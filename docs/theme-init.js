(() => {
  const themeStorageKey = "textbook-vlm-theme";
  const storedTheme = localStorage.getItem(themeStorageKey);
  const mobileDefaultIsDark = window.matchMedia("(max-width: 780px)").matches;
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = storedTheme || (mobileDefaultIsDark || systemPrefersDark ? "dark" : "light");

  document.documentElement.dataset.theme = initialTheme;

  const initialThemeColor = document.querySelector('meta[name="theme-color"]');
  if (initialThemeColor) {
    initialThemeColor.content = initialTheme === "dark" ? "#171716" : "#fbf7ee";
  }
})();
