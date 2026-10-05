// @ts-nocheck
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-gemini-task',
}

// ── LangSmith Tracing Helper ─────────────────────────────────
async function sendLangSmithTrace(data: {
  name: string;
  runType: 'llm' | 'chain' | 'embedding';
  inputs: any;
  outputs?: any;
  error?: string;
  startTime: number;
  endTime: number;
  metadata?: Record<string, any>;
}) {
  try {
    const apiKey = Deno.env.get("LANGCHAIN_API_KEY") || "";
    const endpoint = Deno.env.get("LANGCHAIN_ENDPOINT") || "https://api.smith.langchain.com";
    const projectName = Deno.env.get("LANGCHAIN_PROJECT") || "acadly-chatbot";

    if (!apiKey) return;

    const runId = crypto.randomUUID();

    const payload = {
      id: runId,
      name: data.name,
      run_type: data.runType,
      inputs: data.inputs,
      outputs: data.outputs || null,
      error: data.error || null,
      start_time: new Date(data.startTime).toISOString(),
      end_time: new Date(data.endTime).toISOString(),
      extra: {
        metadata: {
          project_name: projectName,
          ...data.metadata,
        },
      },
    };

    // Non-blocking telemetry post to LangSmith
    fetch(`${endpoint}/runs`, {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }).catch((err) => console.error("LangSmith Telemetry error:", err));
  } catch (err) {
    // Fail silently so LLM execution is never affected
    console.error("LangSmith logging error:", err);
  }
}

serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  const startTime = Date.now();

  try {
    // Parse the full body once
    const body = await req.json() as any;
    const { message, contents, systemPrompt, model, generationConfig, safetySettings, content, requests, taskType } = body;

    // Retrieve API Keys from environment
    // @ts-ignore
    const apiKeyString = Deno.env.get("GEMINI_API_KEYS") || Deno.env.get("GEMINI_API_KEY") || Deno.env.get("API_KEY") || "";
    // @ts-ignore
    const apiKeys = apiKeyString.split(',').map((k: string) => k.trim()).filter((k: string) => k.length > 0);

    if (apiKeys.length === 0) {
      return new Response(
        JSON.stringify({ error: 'MISSING_API_KEY', message: 'Gemini API keys not configured in Supabase secrets.' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
      )
    }

    // 1. Simple request handling
    if (message && !contents) {
      return await handleSimpleChat(message, apiKeys, startTime);
    }

    // 2. Full application request handling
    const selectedModel = model || "gemini-flash-latest";
    const task = req.headers.get("x-gemini-task") || "generateContent";

    // Build payload based on task type
    let payload: any;

    if (task === "generateContent") {
      const defaultGenConfig = {
        maxOutputTokens: 8192,
        temperature: 0.2,
        topP: 0.9,
      };
      payload = {
        system_instruction: systemPrompt ? { parts: [{ text: systemPrompt }] } : undefined,
        contents: contents,
        generationConfig: generationConfig ? { ...defaultGenConfig, ...generationConfig } : defaultGenConfig,
        safetySettings: safetySettings || [
          { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
          { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
          { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH' },
          { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
        ],
      };
    } else if (task === "embedContent") {
      payload = {
        content: content,
        taskType: taskType || 'RETRIEVAL_QUERY',
        outputDimensionality: body.outputDimensionality || 768,
      };
    } else if (task === "batchEmbedContents") {
      // Ensure all batch requests have outputDimensionality set to 768 if not already provided
      const formattedRequests = (requests || []).map((r: any) => ({
        ...r,
        outputDimensionality: r.outputDimensionality || 768,
      }));
      payload = {
        requests: formattedRequests,
      };
    } else {
      const { model: _m, message: _msg, systemPrompt: _sp, generationConfig: _gc, safetySettings: _ss, ...rest } = body;
      payload = rest;
    }

    return await callGeminiWithRetry(selectedModel, task, payload, apiKeys, 0, startTime);

  } catch (err: any) {
    sendLangSmithTrace({
      name: "Supabase_Edge_Function_Error",
      runType: "chain",
      inputs: {},
      error: err?.message || "Unknown error",
      startTime,
      endTime: Date.now(),
    });

    return new Response(
      JSON.stringify({ error: err?.message || 'Unknown error' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
    )
  }
})

/**
 * Handle simple user message -> response
 */
async function handleSimpleChat(message: string, apiKeys: string[], startTime: number) {
  const payload = {
    contents: [{ role: 'user', parts: [{ text: message }] }]
  };
  const res = await callGeminiWithRetry("gemini-flash-latest", "generateContent", payload, apiKeys, 0, startTime);
  const data = await res.clone().json() as any;
  
  if (data.candidates && data.candidates[0]?.content?.parts[0]?.text) {
    return new Response(
      JSON.stringify({ reply: data.candidates[0].content.parts[0].text }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
    );
  }
  
  return res; 
}

/**
 * Call Gemini API with automatic key rotation on 429/403 and trace to LangSmith
 */
async function callGeminiWithRetry(
  model: string, 
  task: string, 
  payload: any, 
  apiKeys: string[], 
  attempt = 0,
  startTime = Date.now()
): Promise<Response> {
  const apiKey = apiKeys[attempt % apiKeys.length];
  // Map older/restricted models to rock-solid stable endpoints automatically
  let actualModel = model;
  if (model === "gemini-2.5-flash" || model === "gemini-2.5-flash-lite" || model === "gemini-flash-latest") {
    actualModel = "gemini-3.5-flash";
  } else if (model === "gemini-2.5-pro" || model === "gemini-pro-latest") {
    actualModel = "gemini-3.5-flash";
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${actualModel}:${task}?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  // If 503 (overloaded) or 404 (unavailable model), fallback to gemini-3.1-flash-lite or retry with key
  if (response.status === 503 && actualModel !== "gemini-3.1-flash-lite") {
    console.log(`Model ${actualModel} hit 503. Falling back to gemini-3.1-flash-lite...`);
    return callGeminiWithRetry("gemini-3.1-flash-lite", task, payload, apiKeys, attempt, startTime);
  }

  if ((response.status === 429 || response.status === 403 || response.status === 503) && attempt < apiKeys.length - 1) {
    console.log(`Key/Request ${attempt} failed with ${response.status}. Rotating...`);
    return callGeminiWithRetry(model, task, payload, apiKeys, attempt + 1, startTime);
  }

  const responseData = await response.json();
  const endTime = Date.now();

  // Send execution trace to LangSmith asynchronously
  sendLangSmithTrace({
    name: task === "generateContent" ? `Gemini:${model}` : `Embedding:${task}`,
    runType: task.includes("embed") ? "embedding" : "llm",
    inputs: payload,
    outputs: responseData,
    error: response.status >= 400 ? JSON.stringify(responseData) : undefined,
    startTime,
    endTime,
    metadata: {
      model,
      task,
      attempts: attempt + 1,
      httpStatus: response.status,
    },
  });

  return new Response(
    JSON.stringify(responseData),
    { 
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }, 
      status: response.status 
    }
  );
}

