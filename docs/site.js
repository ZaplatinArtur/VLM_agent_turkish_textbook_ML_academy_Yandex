const root = document.documentElement;
const themeToggle = document.querySelector(".theme-toggle");
const themeIcon = document.querySelector(".theme-icon");
const menuToggle = document.querySelector(".menu-toggle");
const siteNav = document.querySelector(".site-nav");
const readingProgress = document.querySelector(".reading-progress");
const approachTabs = [...document.querySelectorAll(".approach-tab")];
const approachPanels = [...document.querySelectorAll(".approach-panel")];
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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

const navLinks = siteNav
  ? [...siteNav.querySelectorAll('a[href^="#"]')]
  : [];
const navSections = navLinks
  .map((link) => ({ link, section: document.querySelector(link.getAttribute("href")) }))
  .filter(({ section }) => section);
let scrollFrameRequested = false;

function updateScrollState() {
  const scrollRange = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollRange > 0 ? Math.min(Math.max(window.scrollY / scrollRange, 0), 1) : 0;

  if (readingProgress) {
    root.style.setProperty("--scroll-progress", String(progress));
  }

  if (navSections.length) {
    const marker = window.scrollY + window.innerHeight * 0.35;
    let activeLink = null;

    navSections.forEach(({ link, section }) => {
      if (marker >= section.offsetTop) {
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
  window.addEventListener("scroll", requestScrollStateUpdate, { passive: true });
  window.addEventListener("resize", requestScrollStateUpdate);
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
  };

  selectedPanel.addEventListener("animationend", finishTransition, { once: true });
  window.setTimeout(finishTransition, 360);
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
    { threshold: 0.12 },
  );

  revealElements.forEach((element) => revealObserver.observe(element));
}

function formatCount(value, decimals, suffix) {
  return `${new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value)}${suffix}`;
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
  const duration = 950;

  const previousAnimationFrame = counterAnimationFrames.get(element);
  if (previousAnimationFrame) {
    window.cancelAnimationFrame(previousAnimationFrame);
  }

  function draw(now) {
    const elapsed = Math.min((now - startedAt) / duration, 1);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    element.textContent = formatCount(target * eased, decimals, suffix);

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
