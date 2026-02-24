/**
 * A simple EN / AR language switcher button.
 * Calls onToggle(newLang) when clicked.
 */
export default function LanguageToggle({ lang, onToggle }) {
  const next = lang === "en" ? "ar" : "en";
  const label = lang === "en" ? "العربية" : "English";

  return (
    <button
      onClick={() => onToggle(next)}
      className="text-sm font-medium px-3 py-1.5 rounded-full border border-gray-300
                 hover:bg-gray-100 transition-colors text-gray-700"
      aria-label={`Switch to ${next === "en" ? "English" : "Arabic"}`}
    >
      {label}
    </button>
  );
}
