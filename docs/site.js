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
let themeTransitionInProgress = false;

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
    if (themeTransitionInProgress) {
      return;
    }

    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    themeTransitionInProgress = true;
    themeToggle.setAttribute("aria-busy", "true");

    if (document.startViewTransition && !prefersReducedMotion) {
      const transition = document.startViewTransition(() => setTheme(nextTheme));
      transition.finished.finally(() => {
        themeTransitionInProgress = false;
        themeToggle.removeAttribute("aria-busy");
      });
    } else if (!prefersReducedMotion) {
      root.classList.add("is-theme-fallback");
      window.requestAnimationFrame(() => {
        root.classList.add("is-theme-fade-out");
      });
      window.setTimeout(() => {
        setTheme(nextTheme);
        root.classList.remove("is-theme-fade-out");
      }, 220);
      window.setTimeout(() => {
        root.classList.remove("is-theme-fallback");
        themeTransitionInProgress = false;
        themeToggle.removeAttribute("aria-busy");
      }, 620);
    } else {
      setTheme(nextTheme);
      themeTransitionInProgress = false;
      themeToggle.removeAttribute("aria-busy");
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

let subjectResults = null;

const subjectPicker = document.querySelector("[data-subject-picker]");
const subjectPickerTrigger = subjectPicker?.querySelector(".subject-picker-trigger");
const subjectPickerOptions = [...(subjectPicker?.querySelectorAll("[data-subject-key]") || [])];
const subjectPickerValue = subjectPicker?.querySelector("[data-subject-picker-value]");
const subjectRows = [...document.querySelectorAll("[data-result-mode]")];
const subjectCount = document.getElementById("subject-result-count");
const subjectVerdict = document.getElementById("subject-result-verdict");
const subjectAnimationFrames = new WeakMap();
let currentSubjectResult = null;
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
  const labels = {
    b0: "модель без инструментов",
    text: "текстовый RAG",
    visual: "визуальный RAG",
    hybrid: "гибридный RAG",
  };
  const modes = ["b0", "text", "visual", "hybrid"];
  const bestScore = Math.max(...modes.map((mode) => result[mode]));
  const leaders = modes.filter((mode) => result[mode] === bestScore).map((mode) => labels[mode]);
  const bestRetrievalScore = Math.max(result.text, result.visual, result.hybrid);
  const retrievalDelta = Math.round((bestRetrievalScore - result.b0) * 10) / 10;
  const retrievalComparison = retrievalDelta === 0
    ? "Лучший режим с поиском совпадает с базовой линией."
    : retrievalDelta > 0
      ? `Лучший режим с поиском улучшает результат на ${formatPercent(retrievalDelta).replace("%", " п.п.")}`
      : `Лучший режим с поиском уступает базовой линии на ${formatPercent(Math.abs(retrievalDelta)).replace("%", " п.п.")}`;

  return `${leaders.join(" и ")} ${leaders.length > 1 ? "делят первое место" : "лидирует"}. ${retrievalComparison}`;
}

function updateSubjectResult(key, animate = true) {
  if (!subjectResults?.all) {
    return;
  }

  const result = subjectResults[key] || subjectResults.all;
  const modes = ["b0", "text", "visual", "hybrid"];
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

function initializeSubjectExplorer(results) {
  if (!results?.all || !subjectPicker || !subjectPickerTrigger || !subjectPickerOptions.length || !subjectRows.length) {
    return;
  }

  subjectResults = results;
  currentSubjectResult = subjectResults.all;
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

const visibleCounters = new WeakSet();

function prepareCounter(element) {
  element.classList.add("counter-animated");
  element.setAttribute("aria-label", element.textContent.trim());
}

function setCounterSemanticValue(element, formattedValue) {
  prepareCounter(element);
  element.textContent = formattedValue;
  element.setAttribute("aria-label", formattedValue);
}

function resetCount(element) {
  element.classList.remove("is-count-visible");
}

function animateCount(element) {
  element.classList.remove("is-count-visible");
  void element.offsetWidth;
  element.classList.add("is-count-visible");
}

const counters = [...document.querySelectorAll("[data-count-to]")];

if (prefersReducedMotion) {
  counters.forEach((counter) => {
    setCounterSemanticValue(counter, formatCount(
      Number(counter.dataset.countTo),
      Number(counter.dataset.countDecimals || 0),
      counter.dataset.countSuffix || "",
    ));
  });
} else if (counters.length) {
  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          visibleCounters.add(entry.target);
          animateCount(entry.target);
        } else {
          visibleCounters.delete(entry.target);
          resetCount(entry.target);
        }
      });
    },
    { threshold: 0.35 },
  );

  counters.forEach((counter) => {
    prepareCounter(counter);
    counterObserver.observe(counter);
  });
}

function getResultValue(summary, path) {
  return path.split(".").reduce((value, key) => value?.[key], summary);
}

function formatResultValue(element, value) {
  const decimals = Number(element.dataset.resultDecimals ?? element.dataset.countDecimals ?? 0);
  const prefix = element.dataset.resultPrefix || "";
  const suffix = element.dataset.resultSuffix ?? element.dataset.countSuffix ?? "";
  const number = element.hasAttribute("data-result-absolute") ? Math.abs(Number(value)) : Number(value);
  return `${prefix}${formatCount(number, decimals, suffix)}`;
}

function hydrateResults(summary) {
  document.querySelectorAll("[data-result-key]").forEach((element) => {
    const value = getResultValue(summary, element.dataset.resultKey);
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return;
    }

    const formattedValue = formatResultValue(element, value);
    if (element.hasAttribute("data-count-to")) {
      element.dataset.countTo = String(value);
      setCounterSemanticValue(element, formattedValue);
      if (visibleCounters.has(element)) {
        animateCount(element);
      }
    } else {
      element.textContent = formattedValue;
    }
  });

  document.querySelectorAll("[data-result-width-key]").forEach((element) => {
    const value = getResultValue(summary, element.dataset.resultWidthKey);
    if (typeof value === "number" && Number.isFinite(value)) {
      element.dataset.width = String(value);
    }
  });

  initializeSubjectExplorer(summary.subjects);
}

fetch("data/results-summary.json", { cache: "no-cache" })
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  })
  .then(hydrateResults)
  .catch((error) => {
    console.warn("Не удалось загрузить сводку результатов; показаны значения из HTML.", error);
  });

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
