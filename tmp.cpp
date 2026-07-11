// make_unique / make_shared are preferred over new:
auto p = std::make_unique<Widget>(arg1, arg2);  // ✅
std::unique_ptr<Widget> p(new Widget(arg1, arg2));  // ⚠ avoid
// make_* is exception-safe and often more efficient