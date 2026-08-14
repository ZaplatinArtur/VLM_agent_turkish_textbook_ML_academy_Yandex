const root = document.documentElement;
const themeToggle = document.querySelector(".theme-toggle");
const themeIcon = document.querySelector(".theme-icon");
const menuToggle = document.querySelector(".menu-toggle");
const siteNav = document.querySelector(".site-nav");
const approachTabs = [...document.querySelectorAll(".approach-tab")];
const approachPanels = [...document.querySelectorAll(".approach-panel")];

function setTheme(theme) {
  root.dataset.theme = theme;
  if (themeIcon) {
    themeIcon.textContent = theme === "dark" ? "☾" : "☀";
  }
  if (themeToggle) {
    themeToggle.setAttribute(
      "aria-label",
      theme === "dark" ? "Включить светлую тему" : "Включить тёмную тему",
    );
  }
  const themeColor = document.querySelector('meta[name="theme-color"]');
  if (themeColor) {
    themeColor.content = theme === "dark" ? "#171716" : "#fbf7ee";
  }
}

const storedTheme = localStorage.getItem("textbook-vlm-theme");
const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
  ? "dark"
  : "light";
setTheme(storedTheme || preferredTheme);

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("textbook-vlm-theme", nextTheme);
  });
}

if (menuToggle && siteNav) {
  menuToggle.addEventListener("click", () => {
    const isOpen = siteNav.classList.toggle("is-open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
  });

  siteNav.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      siteNav.classList.remove("is-open");
      menuToggle.setAttribute("aria-expanded", "false");
    }
  });
}

function selectApproach(selectedTab) {
  const selectedApproach = selectedTab.dataset.approach;

  approachTabs.forEach((tab) => {
    const isSelected = tab === selectedTab;
    tab.classList.toggle("is-active", isSelected);
    tab.setAttribute("aria-selected", String(isSelected));
    tab.tabIndex = isSelected ? 0 : -1;
  });

  approachPanels.forEach((panel) => {
    const isSelected = panel.id === `panel-${selectedApproach}`;
    panel.classList.toggle("is-active", isSelected);
    panel.hidden = !isSelected;
  });
}

approachTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectApproach(tab));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowDown', 'ArrowUp', 'ArrowRight', 'ArrowLeft'].includes(event.key)) {
      return;
    }

    event.preventDefault();
    const direction = ['ArrowDown', 'ArrowRight'].includes(event.key) ? 1 : -1;
    const nextIndex = (index + direction + approachTabs.length) % approachTabs.length;
    approachTabs[nextIndex].focus();
    selectApproach(approachTabs[nextIndex]);
  });
});

const revealObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }

      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  },
  { threshold: 0.12 },
);

document.querySelectorAll(".reveal").forEach((element) => {
  revealObserver.observe(element);
});

const barObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) {
        return;
      }

      entry.target.style.width = `${entry.target.dataset.width}%`;
      observer.unobserve(entry.target);
    });
  },
  { threshold: 0.5 },
);

document.querySelectorAll(".bar").forEach((bar) => barObserver.observe(bar));

const year = document.getElementById("year");
if (year) {
  year.textContent = String(new Date().getFullYear());
}
