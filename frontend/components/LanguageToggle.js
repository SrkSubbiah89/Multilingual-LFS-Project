/**
 * Multi-language selector — supports EN, AR, UR, HI, TL.
 * Calls onToggle(newLang) when the user picks a different language.
 */

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "ar", label: "العربية" },
  { code: "ur", label: "اردو" },
  { code: "hi", label: "हिन्दी" },
  { code: "tl", label: "Filipino" },
];

export default function LanguageToggle({ lang, onToggle, languages }) {
  // ar-gulf is a backend dialect tag — display it as Arabic in the selector
  const displayLang = lang === "ar-gulf" ? "ar" : lang;
  const isRtl = displayLang === "ar" || displayLang === "ur";
  // Optional `languages` prop restricts the offered options to a subset
  // (e.g. ["en", "ar"]) for a page that genuinely only supports those —
  // added 2026-08-29 so a page can't offer a language selection that
  // silently does nothing once picked. Defaults to all 5 supported languages.
  const options = languages
    ? LANGUAGES.filter((l) => languages.includes(l.code))
    : LANGUAGES;

  return (
    <select
      value={displayLang}
      onChange={(e) => onToggle(e.target.value)}
      dir={isRtl ? "rtl" : "ltr"}
      className="text-sm font-medium px-3 py-1.5 rounded-full border border-gray-300
                 hover:bg-gray-100 transition-colors text-gray-700 bg-white cursor-pointer"
      aria-label="Select language"
    >
      {options.map(({ code, label }) => (
        <option key={code} value={code}>
          {label}
        </option>
      ))}
    </select>
  );
}
