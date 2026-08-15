const root = document.documentElement;
const themeToggle = document.querySelector(".theme-toggle");
const themeIcon = document.querySelector(".theme-icon");
const menuToggle = document.querySelector(".menu-toggle");
const siteNav = document.querySelector(".site-nav");
const readingProgress = document.querySelector(".reading-progress");
const approachTabs = [...document.querySelectorAll(".approach-tab")];
const approachPanels = [...document.querySelectorAll(".approach-panel")];
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const mobileNavigation = window.matchMedia("(max-width: 900px)");
let mobileScrollFrame = null;
let previousScrollBehavior = "";

function scrollToMobileSection(target) {
  const startY = window.scrollY;
  const targetY = Math.max(
    0,
    target.getBoundingClientRect().top + startY - 88,
  );
  const distance = targetY - startY;
  const duration = Math.min(1300, Math.max(700, Math.abs(distance) * 0.2));
  const startedAt = performance.now();

  if (mobileScrollFrame) {
    window.cancelAnimationFrame(mobileScrollFrame);
  } else {
    previousScrollBehavior = root.style.scrollBehavior;
  }

  root.style.scrollBehavior = "auto";

  function animateScroll(now) {
    const progress = Math.min((now - startedAt) / duration, 1);
    const easedProgress =
      progress < 0.5
        ? 4 * progress * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 3) / 2;

    window.scrollTo(0, startY + distance * easedProgress);

    if (progress < 1) {
      mobileScrollFrame = window.requestAnimationFrame(animateScroll);
    } else {
      mobileScrollFrame = null;
      root.style.scrollBehavior = previousScrollBehavior;
    }
  }

  mobileScrollFrame = window.requestAnimationFrame(animateScroll);
}

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
    themeColor.content = theme === "dark" ? "#15130f" : "#efe8d8";
  }
}

const storedTheme = localStorage.getItem("textbook-vlm-theme");
const isMobileViewport = window.matchMedia("(max-width: 780px)").matches;
const preferredTheme = isMobileViewport
  ? "dark"
  : window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
setTheme(storedTheme || preferredTheme);

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";

    if (document.startViewTransition && !prefersReducedMotion) {
      document.startViewTransition(() => setTheme(nextTheme));
    } else if (!prefersReducedMotion) {
      root.classList.add("is-theme-fallback", "is-theme-fade-out");
      window.setTimeout(() => {
        setTheme(nextTheme);
        root.classList.remove("is-theme-fade-out");
      }, 100);
      window.setTimeout(() => root.classList.remove("is-theme-fallback"), 260);
    } else {
      setTheme(nextTheme);
    }

    localStorage.setItem("textbook-vlm-theme", nextTheme);
  });
}

if (menuToggle && siteNav) {
  menuToggle.addEventListener("click", () => {
    const isOpen = siteNav.classList.toggle("is-open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
  });

  siteNav.addEventListener("click", (event) => {
    const link = event.target.closest('a[href^="#"]');

    if (!link) {
      return;
    }

    siteNav.classList.remove("is-open");
    menuToggle.setAttribute("aria-expanded", "false");

    if (!mobileNavigation.matches || prefersReducedMotion) {
      return;
    }

    const target = document.querySelector(link.getAttribute("href"));

    if (!target) {
      return;
    }

    event.preventDefault();
    window.history.pushState(null, "", link.hash);
    scrollToMobileSection(target);
  });
}

const navLinks = siteNav
  ? [...siteNav.querySelectorAll('a[href^="#"]')]
  : [];
const navSections = navLinks
  .map((link) => ({ link, section: document.querySelector(link.getAttribute("href")) }))
  .filter(({ section }) => section);
let navSectionTops = [];
let scrollFrameRequested = false;

function cacheNavSectionTops() {
  navSectionTops = navSections.map(({ link, section }) => ({
    link,
    top: section.getBoundingClientRect().top + window.scrollY,
  }));
}

function updateScrollState() {
  const scrollRange = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollRange > 0 ? Math.min(Math.max(window.scrollY / scrollRange, 0), 1) : 0;

  if (readingProgress) {
    root.style.setProperty("--scroll-progress", String(progress));
  }

  if (navSectionTops.length) {
    const marker = window.scrollY + window.innerHeight * 0.35;
    let activeLink = null;

    navSectionTops.forEach(({ link, top }) => {
      if (marker >= top) {
        activeLink = link;
      }
    });

    navLinks.forEach((link) => {
      const isActive = link === activeLink;
      link.classList.toggle("is-active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  scrollFrameRequested = false;
}

function requestScrollStateUpdate() {
  if (scrollFrameRequested) {
    return;
  }

  scrollFrameRequested = true;
  window.requestAnimationFrame(updateScrollState);
}

if (readingProgress || navSections.length) {
  cacheNavSectionTops();
  window.addEventListener("scroll", requestScrollStateUpdate, { passive: true });
  window.addEventListener("resize", () => {
    cacheNavSectionTops();
    requestScrollStateUpdate();
  }, { passive: true });
  window.addEventListener("load", cacheNavSectionTops, { once: true });
  document.fonts?.ready.then(cacheNavSectionTops);
  requestScrollStateUpdate();
}

function selectApproach(selectedTab) {
  const selectedApproach = selectedTab.dataset.approach;
  const selectedPanel = approachPanels.find(
    (panel) => panel.id === `panel-${selectedApproach}`,
  );
  const activePanel = approachPanels.find((panel) => panel.classList.contains("is-active"));

  if (!selectedPanel || selectedPanel === activePanel) {
    return;
  }

  approachTabs.forEach((tab) => {
    const isSelected = tab === selectedTab;
    tab.classList.toggle("is-active", isSelected);
    tab.setAttribute("aria-selected", String(isSelected));
    tab.tabIndex = isSelected ? 0 : -1;
  });

  if (prefersReducedMotion || !activePanel) {
    approachPanels.forEach((panel) => {
      const isSelected = panel === selectedPanel;
      panel.classList.toggle("is-active", isSelected);
      panel.classList.remove("is-entering", "is-leaving");
      panel.hidden = !isSelected;
      panel.setAttribute("aria-hidden", String(!isSelected));
    });
    cacheNavSectionTops();
    requestScrollStateUpdate();
    return;
  }

  approachPanels.forEach((panel) => {
    if (panel !== activePanel && panel !== selectedPanel) {
      panel.hidden = true;
      panel.classList.remove("is-active", "is-entering", "is-leaving");
      panel.setAttribute("aria-hidden", "true");
    }
  });

  activePanel.classList.remove("is-active", "is-entering");
  activePanel.classList.add("is-leaving");
  activePanel.setAttribute("aria-hidden", "true");

  selectedPanel.hidden = false;
  selectedPanel.setAttribute("aria-hidden", "false");
  selectedPanel.classList.remove("is-entering", "is-leaving");
  void selectedPanel.offsetWidth;
  selectedPanel.classList.add("is-active", "is-entering");

  let transitionFinished = false;
  const finishTransition = (event) => {
    if (event && event.target !== selectedPanel) {
      return;
    }
    if (transitionFinished) {
      return;
    }

    transitionFinished = true;
    if (!activePanel.classList.contains("is-active")) {
      activePanel.hidden = true;
    }
    activePanel.classList.remove("is-leaving");
    selectedPanel.classList.remove("is-entering");
    cacheNavSectionTops();
    requestScrollStateUpdate();
  };

  selectedPanel.addEventListener("animationend", finishTransition, { once: true });
  window.setTimeout(finishTransition, 600);
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

const revealElements = [...document.querySelectorAll(".reveal")];

if (prefersReducedMotion) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else if (revealElements.length) {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        entry.target.classList.toggle("is-visible", entry.isIntersecting);
      });
    },
    { rootMargin: "0px 0px -6% 0px", threshold: 0.01 },
  );

  revealElements.forEach((element) => revealObserver.observe(element));
}

const subjectResults = {
  all: { name: "Все предметы", count: 198, b0: 57.1, web: 70.7, rag: 54.5 },
  ataturk: { name: "Ататюркизм", count: 1, b0: 0, web: 100, rag: 100 },
  biology: { name: "Биология", count: 18, b0: 44.4, web: 61.1, rag: 44.4 },
  chemistry: { name: "Химия", count: 32, b0: 62.5, web: 81.2, rag: 59.4 },
  english: { name: "Английский", count: 9, b0: 88.9, web: 88.9, rag: 77.8 },
  geography: { name: "География", count: 14, b0: 57.1, web: 71.4, rag: 78.6 },
  history: { name: "История", count: 10, b0: 50, web: 70, rag: 60 },
  math: { name: "Математика", count: 64, b0: 51.6, web: 73.4, rag: 46.9 },
  philosophy: { name: "Философия", count: 2, b0: 50, web: 50, rag: 100 },
  physics: { name: "Физика", count: 19, b0: 73.7, web: 73.7, rag: 57.9 },
  science: { name: "Естествознание", count: 5, b0: 40, web: 20, rag: 40 },
  sociology: { name: "Социология", count: 3, b0: 66.7, web: 66.7, rag: 66.7 },
  turkish: { name: "Турецкий язык и литература", count: 21, b0: 57.1, web: 57.1, rag: 42.9 },
};

const subjectPicker = document.querySelector("[data-subject-picker]");
const subjectPickerTrigger = subjectPicker?.querySelector(".subject-picker-trigger");
const subjectPickerOptions = [...(subjectPicker?.querySelectorAll("[data-subject-key]") || [])];
const subjectPickerValue = subjectPicker?.querySelector("[data-subject-picker-value]");
const subjectRows = [...document.querySelectorAll("[data-result-mode]")];
const subjectCount = document.getElementById("subject-result-count");
const subjectVerdict = document.getElementById("subject-result-verdict");
const subjectAnimationFrames = new WeakMap();
let currentSubjectResult = subjectResults.all;
const numberFormatters = new Map();

function getNumberFormatter(decimals) {
  const key = String(decimals);
  if (!numberFormatters.has(key)) {
    numberFormatters.set(key, new Intl.NumberFormat("ru-RU", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }));
  }
  return numberFormatters.get(key);
}

function formatPercent(value) {
  return `${getNumberFormatter(Number.isInteger(value) ? 0 : 1).format(value)}%`;
}

function animateSubjectValue(element, from, to) {
  const previousFrame = subjectAnimationFrames.get(element);
  if (previousFrame) {
    window.cancelAnimationFrame(previousFrame);
  }

  if (prefersReducedMotion || from === to) {
    element.textContent = formatPercent(to);
    return;
  }

  const startedAt = performance.now();
  const duration = 1050;
  let lastPaint = 0;

  function draw(now) {
    const progress = Math.min((now - startedAt) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    if (now - lastPaint >= 32 || progress === 1) {
      element.textContent = formatPercent(from + (to - from) * eased);
      lastPaint = now;
    }

    if (progress < 1) {
      subjectAnimationFrames.set(element, window.requestAnimationFrame(draw));
    } else {
      element.textContent = formatPercent(to);
      subjectAnimationFrames.delete(element);
    }
  }

  subjectAnimationFrames.set(element, window.requestAnimationFrame(draw));
}

function describeSubjectResult(result) {
  const labels = { b0: "модель без инструментов", web: "веб-поиск", rag: "Textbook RAG" };
  const modes = ["b0", "web", "rag"];
  const bestScore = Math.max(...modes.map((mode) => result[mode]));
  const leaders = modes.filter((mode) => result[mode] === bestScore).map((mode) => labels[mode]);
  const ragDelta = Math.round((result.rag - result.b0) * 10) / 10;
  const ragComparison = ragDelta === 0
    ? "RAG не меняет точность относительно режима без инструментов."
    : ragDelta > 0
      ? `RAG улучшает результат на ${formatPercent(ragDelta).replace("%", " п.п.")}`
      : `RAG уступает режиму без инструментов на ${formatPercent(Math.abs(ragDelta)).replace("%", " п.п.")}`;

  return `${leaders.join(" и ")} ${leaders.length > 1 ? "делят первое место" : "лидирует"}. ${ragComparison}`;
}

function updateSubjectResult(key, animate = true) {
  const result = subjectResults[key] || subjectResults.all;
  const modes = ["b0", "web", "rag"];
  const bestScore = Math.max(...modes.map((mode) => result[mode]));

  subjectRows.forEach((row) => {
    const mode = row.dataset.resultMode;
    const valueElement = row.querySelector("[data-subject-value]");
    const bar = row.querySelector("[data-subject-bar]");
    const fromValue = currentSubjectResult[mode];

    row.classList.toggle("is-best", result[mode] === bestScore);
    if (bar) {
      if (animate && !prefersReducedMotion) {
        bar.style.width = `${fromValue}%`;
        window.requestAnimationFrame(() => {
          bar.style.width = `${result[mode]}%`;
        });
      } else {
        bar.style.width = `${result[mode]}%`;
      }
    }
    if (valueElement) {
      animateSubjectValue(valueElement, animate ? fromValue : result[mode], result[mode]);
    }
  });

  if (subjectCount) {
    subjectCount.textContent = `${result.count} ${result.count === 1 ? "задание" : "заданий"}`;
  }
  if (subjectVerdict) {
    subjectVerdict.textContent = describeSubjectResult(result);
  }
  currentSubjectResult = result;
}

function setSubjectPickerOpen(isOpen) {
  if (!subjectPicker || !subjectPickerTrigger) {
    return;
  }
  subjectPicker.classList.toggle("is-open", isOpen);
  subjectPickerTrigger.setAttribute("aria-expanded", String(isOpen));
  const options = subjectPicker.querySelector(".subject-picker-options");
  if (options) {
    options.hidden = !isOpen;
  }
}

function chooseSubject(option) {
  const key = option.dataset.subjectKey;
  const result = subjectResults[key] || subjectResults.all;
  subjectPickerOptions.forEach((item) => {
    item.setAttribute("aria-selected", String(item === option));
  });
  if (subjectPickerValue) {
    subjectPickerValue.textContent = result.name;
  }
  updateSubjectResult(key);
  setSubjectPickerOpen(false);
  subjectPickerTrigger?.focus();
}

if (subjectPicker && subjectPickerTrigger && subjectPickerOptions.length && subjectRows.length) {
  updateSubjectResult("all", false);
  subjectPickerTrigger.addEventListener("click", () => {
    const willOpen = subjectPickerTrigger.getAttribute("aria-expanded") !== "true";
    setSubjectPickerOpen(willOpen);
    if (willOpen) {
      (subjectPickerOptions.find((option) => option.getAttribute("aria-selected") === "true") || subjectPickerOptions[0]).focus();
    }
  });
  subjectPickerTrigger.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    setSubjectPickerOpen(true);
    const selectedIndex = Math.max(0, subjectPickerOptions.findIndex((option) => option.getAttribute("aria-selected") === "true"));
    const targetIndex = event.key === "ArrowDown"
      ? selectedIndex
      : (selectedIndex - 1 + subjectPickerOptions.length) % subjectPickerOptions.length;
    subjectPickerOptions[targetIndex].focus();
  });
  subjectPickerOptions.forEach((option, index) => {
    option.addEventListener("click", () => chooseSubject(option));
    option.addEventListener("keydown", (event) => {
      if (!["ArrowDown", "ArrowUp", "Home", "End", "Escape"].includes(event.key)) {
        return;
      }
      event.preventDefault();
      if (event.key === "Escape") {
        setSubjectPickerOpen(false);
        subjectPickerTrigger.focus();
        return;
      }
      const nextIndex = event.key === "Home"
        ? 0
        : event.key === "End"
          ? subjectPickerOptions.length - 1
          : (index + (event.key === "ArrowDown" ? 1 : -1) + subjectPickerOptions.length) % subjectPickerOptions.length;
      subjectPickerOptions[nextIndex].focus();
    });
  });
  document.addEventListener("click", (event) => {
    if (!subjectPicker.contains(event.target)) {
      setSubjectPickerOpen(false);
    }
  });
}

function formatCount(value, decimals, suffix) {
  return `${getNumberFormatter(decimals).format(value)}${suffix}`;
}

const counterAnimationFrames = new WeakMap();

function resetCount(element) {
  const animationFrame = counterAnimationFrames.get(element);
  if (animationFrame) {
    window.cancelAnimationFrame(animationFrame);
    counterAnimationFrames.delete(element);
  }

  element.textContent = formatCount(
    0,
    Number(element.dataset.countDecimals || 0),
    element.dataset.countSuffix || "",
  );
}

function animateCount(element) {
  const target = Number(element.dataset.countTo);
  const decimals = Number(element.dataset.countDecimals || 0);
  const suffix = element.dataset.countSuffix || "";
  const startedAt = performance.now();
  const duration = 1400;
  let lastPaint = 0;

  const previousAnimationFrame = counterAnimationFrames.get(element);
  if (previousAnimationFrame) {
    window.cancelAnimationFrame(previousAnimationFrame);
  }

  function draw(now) {
    const elapsed = Math.min((now - startedAt) / duration, 1);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    if (now - lastPaint >= 32 || elapsed === 1) {
      element.textContent = formatCount(target * eased, decimals, suffix);
      lastPaint = now;
    }

    if (elapsed < 1) {
      counterAnimationFrames.set(element, window.requestAnimationFrame(draw));
    } else {
      element.textContent = formatCount(target, decimals, suffix);
      counterAnimationFrames.delete(element);
    }
  }

  counterAnimationFrames.set(element, window.requestAnimationFrame(draw));
}

const counters = [...document.querySelectorAll("[data-count-to]")];

if (prefersReducedMotion) {
  counters.forEach((counter) => {
    counter.textContent = formatCount(
      Number(counter.dataset.countTo),
      Number(counter.dataset.countDecimals || 0),
      counter.dataset.countSuffix || "",
    );
  });
} else if (counters.length) {
  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCount(entry.target);
        } else {
          resetCount(entry.target);
        }
      });
    },
    { threshold: 0.35 },
  );

  counters.forEach((counter) => {
    counter.setAttribute("aria-label", counter.textContent.trim());
    resetCount(counter);
    counterObserver.observe(counter);
  });
}

const chartCards = [...document.querySelectorAll(".chart-card")];

function fillChartBars(card) {
  card.querySelectorAll(".bar").forEach((bar) => {
    bar.style.width = `${bar.dataset.width}%`;
  });
}

function resetChartBars(card) {
  card.querySelectorAll(".bar").forEach((bar) => {
    bar.style.width = "0%";
  });
}

if (prefersReducedMotion) {
  chartCards.forEach(fillChartBars);
} else if (chartCards.length) {
  const chartObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          fillChartBars(entry.target);
        } else {
          resetChartBars(entry.target);
        }
      });
    },
    { threshold: 0.22 },
  );

  chartCards.forEach((card) => chartObserver.observe(card));
}

const systemMap = document.querySelector(".system-map");

if (systemMap && !prefersReducedMotion) {
  systemMap.classList.add("is-motion-pending");
  const systemObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        entry.target.classList.toggle("is-animated", entry.isIntersecting);
      });
    },
    { threshold: 0.28 },
  );

  systemObserver.observe(systemMap);
}

const year = document.getElementById("year");
if (year) {
  year.textContent = String(new Date().getFullYear());
}
