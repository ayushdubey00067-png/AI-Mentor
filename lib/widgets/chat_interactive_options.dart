// lib/widgets/chat_interactive_options.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../utils/app_theme.dart';

class InteractiveOptionParser {
  /// Extracts interactive options from AI message content.
  /// Looks for [OPTIONS: opt1 | opt2] tags, or numbered "Next Action:" lines.
  static ParsedAiMessage parse(String content) {
    String cleanContent = content;
    final List<String> options = [];

    // 1. Explicit [OPTIONS: opt1 | opt2 | opt3] tag
    final optionsTagMatch = RegExp(r'\[OPTIONS:\s*([^\]]+)\]', caseSensitive: false).firstMatch(content);
    if (optionsTagMatch != null) {
      final rawOpts = optionsTagMatch.group(1) ?? '';
      final splitOpts = rawOpts.split('|').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();
      options.addAll(splitOpts);
      cleanContent = cleanContent.replaceAll(optionsTagMatch.group(0)!, '').trim();
    }

    // 2. Parse numbered "Next Action:" or "Would you like to:" bullet points if no explicit tag
    if (options.isEmpty) {
      final nextActionMatch = RegExp(
        r'(?:Next Action|Suggested Next Action|Would you like to|Options):\s*([\s\S]+)$',
        caseSensitive: false,
      ).firstMatch(content);

      if (nextActionMatch != null) {
        final actionBlock = nextActionMatch.group(1) ?? '';
        final lineMatches = RegExp(r'(?:^\s*\d+[\.\)]\s*|\n\s*\d+[\.\)]\s*|\n\s*[-*•]\s*)([^\n]+)')
            .allMatches(actionBlock);

        for (final m in lineMatches) {
          final opt = m.group(1)?.trim() ?? '';
          // Clean out leading punctuation or formatting
          final cleanOpt = opt
              .replaceAll(RegExp(r'^\*+|\*+$'), '')
              .replaceAll(RegExp(r'^_+|_+$'), '')
              .trim();
          if (cleanOpt.length > 3 && cleanOpt.length < 90) {
            options.add(cleanOpt);
          }
        }
      }
    }

    // 3. Fallback: Parse inline question choices like "either X or Y" or candidate student matches
    if (options.isEmpty && content.toLowerCase().contains('did you mean') || content.toLowerCase().contains('select from')) {
      final candidateMatches = RegExp(r'(?:[-*•]|\d+\.)\s*([A-Z0-9\s]+(?:\([^\)]+\))?)').allMatches(content);
      for (final m in candidateMatches) {
        final opt = m.group(1)?.trim() ?? '';
        if (opt.length > 3 && opt.length < 80) {
          options.add(opt);
        }
      }
    }

    return ParsedAiMessage(
      cleanText: cleanContent,
      options: options.toSet().toList(),
    );
  }
}

class ParsedAiMessage {
  final String cleanText;
  final List<String> options;
  ParsedAiMessage({required this.cleanText, required this.options});
}

class ChatInteractiveOptionsView extends StatelessWidget {
  final List<String> options;
  final void Function(String selectedOption) onOptionSelected;
  final Color primaryColor;

  const ChatInteractiveOptionsView({
    super.key,
    required this.options,
    required this.onOptionSelected,
    this.primaryColor = AppTheme.mentorBubble,
  });

  IconData _getIconForOption(String text) {
    final lower = text.toLowerCase();
    if (lower.contains('next') || lower.contains('more') || lower.contains('forward')) {
      return Icons.arrow_forward_rounded;
    }
    if (lower.contains('prev') || lower.contains('back')) {
      return Icons.arrow_back_rounded;
    }
    if (lower.contains('top') || lower.contains('distinction') || lower.contains('performer') || lower.contains('highest')) {
      return Icons.stars_rounded;
    }
    if (lower.contains('fail') || lower.contains('backlog') || lower.contains('at-risk') || lower.contains('risk') || lower.contains('warning')) {
      return Icons.warning_amber_rounded;
    }
    if (lower.contains('search') || lower.contains('find') || lower.contains('lookup')) {
      return Icons.search_rounded;
    }
    if (lower.contains('nikhil') || lower.contains('student') || lower.contains('aayush') || lower.contains('aditya') || lower.contains('roll')) {
      return Icons.person_rounded;
    }
    if (lower.contains('result') || lower.contains('mark') || lower.contains('grade') || lower.contains('sgpa')) {
      return Icons.assessment_rounded;
    }
    if (lower.contains('attendance') || lower.contains('presence')) {
      return Icons.fact_check_rounded;
    }
    if (lower.contains('calendar') || lower.contains('date') || lower.contains('schedule')) {
      return Icons.calendar_today_rounded;
    }
    if (lower.contains('all') || lower.contains('list') || lower.contains('overview')) {
      return Icons.view_list_rounded;
    }
    return Icons.touch_app_rounded;
  }

  @override
  Widget build(BuildContext context) {
    if (options.isEmpty) return const SizedBox.shrink();

    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.auto_awesome_rounded, size: 14, color: primaryColor),
              const SizedBox(width: 6),
              Text(
                'Quick Selection / Action:',
                style: GoogleFonts.lato(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: primaryColor,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: options.map((opt) {
              return Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: () => onOptionSelected(opt),
                  borderRadius: BorderRadius.circular(20),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                    decoration: BoxDecoration(
                      color: primaryColor.withOpacity(0.08),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: primaryColor.withOpacity(0.3),
                        width: 1.2,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: primaryColor.withOpacity(0.04),
                          blurRadius: 4,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _getIconForOption(opt),
                          size: 14,
                          color: primaryColor,
                        ),
                        const SizedBox(width: 6),
                        Flexible(
                          child: Text(
                            opt,
                            style: GoogleFonts.lato(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: primaryColor,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}
