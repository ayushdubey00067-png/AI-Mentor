// lib/services/auth_provider.dart
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import '../models/models.dart';
import 'supabase_service.dart';

class AuthProvider extends ChangeNotifier {
  UserModel? _currentUser;
  bool _isLoading     = false;
  bool _sessionChecked = false;

  UserModel? get currentUser    => _currentUser;
  bool get isLoading            => _isLoading;
  bool get isLoggedIn           => _currentUser != null;
  bool get isMentor             => _currentUser?.isMentor  ?? false;
  bool get isStudent            => _currentUser?.isStudent ?? false;
  bool get sessionChecked       => _sessionChecked;

  AuthProvider() { _restoreSession(); }

  Future<void> _restoreSession() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final json  = prefs.getString('session_user');
      if (json != null) {
        _currentUser = UserModel.fromMap(jsonDecode(json) as Map<String, dynamic>);
        debugPrint('♻️ Session restored: ${_currentUser!.email}');
      }
    } catch (e) {
      debugPrint('⚠️ Session restore failed: $e');
      _currentUser = null;
    } finally {
      _sessionChecked = true;
      notifyListeners();
    }
  }

  Future<void> _save(UserModel u) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('session_user', jsonEncode(u.toMap()));
  }

  Future<void> _clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('session_user');
  }

  Future<String?> login(String email, String password) async {
    _isLoading = true; notifyListeners();
    try {
      final user = await SupabaseService.login(email, password);
      if (user == null) return 'invalid_credentials';
      _currentUser = user;
      await _save(user);
      return null;
    } catch (e) {
      return e.toString();
    } finally {
      _isLoading = false; notifyListeners();
    }
  }

  Future<String?> register({
    required String email,
    required String password,
    required String name,
    required String role,
    String? program,
    String? branch,
    String? semester,
    String? mentorEmail,
    String? rollNumber,
    String? department,
    String? designation,
    String? phone,
  }) async {
    _isLoading = true; notifyListeners();
    try {
      final user = await SupabaseService.register(
        email: email, password: password, name: name, role: role,
        program: program, branch: branch, semester: semester,
        mentorEmail: mentorEmail, rollNumber: rollNumber,
        department: department, designation: designation, phone: phone,
      );
      _currentUser = user;
      await _save(user);
      return null;
    } catch (e) {
      return e.toString();
    } finally {
      _isLoading = false; notifyListeners();
    }
  }

  Future<void> logout() async {
    _currentUser = null;
    await _clear();
    notifyListeners();
  }
}