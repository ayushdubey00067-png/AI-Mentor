// @ts-nocheck
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-gemini-task',
}


serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Parse the full body once
    const body = await req.json() as any;
    const { message, contents, systemPrompt, model, generationConfig, safetySettings, content, requests, taskType } = body;

    // Retrieve API Keys from environment
    // @ts-ignore
    const apiKeyString = Deno.env.get("GEMINI_API_KEYS") || Deno.env.get("API_KEY") || "";
    // @ts-ignore
    const apiKeys = apiKeyString.split(',').map((k: string) => k.trim()).filter((k: string) => k.length > 0);


    if (apiKeys.length === 0) {
      return new Response(
        JSON.stringify({ error: 'MISSING_API_KEY', message: 'Gemini API keys not configured in Supabase secrets.' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
      )
    }

    // 1. Simple request handling (as per user instruction)
    if (message && !contents) {
      return await handleSimpleChat(message, apiKeys);
    }

    // 2. Full application request handling
    const selectedModel = model || "gemini-2.5-flash-lite";
    const task = req.headers.get("x-gemini-task") || "generateContent";

    // Build payload based on task type
    let payload: any;

    if (task === "generateContent") {
      // Chat/generation tasks — include system prompt, safety, config
      payload = {
        system_instruction: systemPrompt ? { parts: [{ text: systemPrompt }] } : undefined,
        contents: contents,
        generationConfig: generationConfig || {
          maxOutputTokens: 2048,
          temperature: 0.7,
          topP: 0.9,
        },
        safetySettings: safetySettings || [
          { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
          { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
          { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_ONLY_HIGH' },
          { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
        ],
      };
    } else if (task === "embedContent") {
      // Single embedding — only content + taskType allowed
      payload = {
        content: content,
        taskType: taskType || 'RETRIEVAL_QUERY',
      };
    } else if (task === "batchEmbedContents") {
      // Batch embeddings — only requests array allowed
      payload = {
        requests: requests,
      };
    } else {
      // Unknown task — pass body without generationConfig/safetySettings
      const { model: _m, message: _msg, systemPrompt: _sp, generationConfig: _gc, safetySettings: _ss, ...rest } = body;
      payload = rest;
    }

    return await callGeminiWithRetry(selectedModel, task, payload, apiKeys);

  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: err?.message || 'Unknown error' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
    )
  }
})


/**
 * Handle simple user message -> response
 */
async function handleSimpleChat(message: string, apiKeys: string[]) {
  const payload = {
    contents: [{ role: 'user', parts: [{ text: message }] }]
  };
  const res = await callGeminiWithRetry("gemini-2.5-flash", "generateContent", payload, apiKeys);
  const data = await res.json() as any;
  
  if (data.candidates && data.candidates[0]?.content?.parts[0]?.text) {
    return new Response(
      JSON.stringify({ reply: data.candidates[0].content.parts[0].text }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
    );
  }
  
  return res; 
}

/**
 * Call Gemini API with automatic key rotation on 429/403
 */
async function callGeminiWithRetry(model: string, task: string, payload: any, apiKeys: string[], attempt = 0): Promise<Response> {
  const apiKey = apiKeys[attempt % apiKeys.length];
  // task can be generateContent, embedContent, batchEmbedContents, etc.
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:${task}?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if ((response.status === 429 || response.status === 403) && attempt < apiKeys.length - 1) {
    console.log(`Key ${attempt} failed with ${response.status}. Rotating...`);
    return callGeminiWithRetry(model, task, payload, apiKeys, attempt + 1);
  }

  const responseData = await response.json();
  return new Response(
    JSON.stringify(responseData),
    { 
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }, 
      status: response.status 
    }
  );
}
