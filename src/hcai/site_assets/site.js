/* Interface language switch (EN / PT-PT) and theme switch (auto / light / dark). Only interface strings
   ([data-i18n], and aria-labels named by [data-i18n-label]) change language; repository prose stays English.
   Both choices are remembered in localStorage when the browser allows it. The stored theme is applied by the
   inline script in each page's <head>, before the first paint; this file only changes it. */
(function () {
  "use strict";
  var STRINGS = /*STRINGS*/null;
  var KEY = "hcai-lang";
  var THEME_KEY = "hcai-theme";
  var THEMES = ["auto", "light", "dark"];

  function stored(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function store(key, value) {
    try {
      if (value === null) { window.localStorage.removeItem(key); } else { window.localStorage.setItem(key, value); }
    } catch (e) { /* not saved: the page still switches */ }
  }

  var current = stored(KEY) === "pt" ? "pt" : "en";

  function t(key, lang) {
    var table = STRINGS && STRINGS[lang || current];
    return (table && table[key]) || (STRINGS && STRINGS.en[key]) || key;
  }

  function apply(lang) {
    current = lang;
    var nodes = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      node.textContent = t(node.getAttribute("data-i18n"), lang);
      if (lang === "pt") { node.setAttribute("lang", "pt"); } else { node.removeAttribute("lang"); }
    }
    var labelled = document.querySelectorAll("[data-i18n-label]");
    for (var k = 0; k < labelled.length; k++) {
      labelled[k].setAttribute("aria-label", t(labelled[k].getAttribute("data-i18n-label"), lang));
    }
    var button = document.getElementById("lang-toggle");
    if (button) {
      var other = lang === "pt" ? "en" : "pt";
      button.textContent = t("lang.switch", lang);
      button.setAttribute("lang", other);
      button.setAttribute("aria-label", t("lang.switch.label", lang));
    }
    document.dispatchEvent(new CustomEvent("hcai:lang", { detail: lang }));
  }

  /* The theme: "auto" follows the system (no attribute); "light" and "dark" set data-theme on <html>. */
  function theme() {
    var value = document.documentElement.getAttribute("data-theme");
    return value === "light" || value === "dark" ? value : "auto";
  }
  function showTheme(button) {
    button.setAttribute("data-i18n", "theme." + theme());
    button.textContent = t("theme." + theme());
    if (current === "pt") { button.setAttribute("lang", "pt"); } else { button.removeAttribute("lang"); }
  }
  function setTheme(value) {
    if (value === "auto") {
      document.documentElement.removeAttribute("data-theme");
      store(THEME_KEY, null);
    } else {
      document.documentElement.setAttribute("data-theme", value);
      store(THEME_KEY, value);
    }
  }

  window.hcaiT = function (key) { return t(key); };

  function start() {
    var themeButton = document.getElementById("theme-toggle");
    if (themeButton) {
      showTheme(themeButton);
      themeButton.addEventListener("click", function () {
        setTheme(THEMES[(THEMES.indexOf(theme()) + 1) % THEMES.length]);
        showTheme(themeButton);
      });
    }
    if (current !== "en") { apply(current); }
    var button = document.getElementById("lang-toggle");
    if (button) {
      button.addEventListener("click", function () {
        var next = current === "pt" ? "en" : "pt";
        store(KEY, next);
        apply(next);
      });
    }
    // On a phone the nav is one row that scrolls sideways and fades at its right edge (style.css). Keep the
    // current page's link, and any link that takes the focus, whole and clear of the fade: a browser does not
    // scroll a link that is only partly hidden.
    var row = document.querySelector(".site-nav ul");
    var FADE = 40;  // the fade's width: 2.5rem
    function reveal(link) {
      if (!row || row.scrollWidth <= row.clientWidth) { return; }
      var l = link.getBoundingClientRect();
      var r = row.getBoundingClientRect();
      if (l.right > r.right - FADE) {
        row.scrollLeft += l.right - (r.right - FADE);
      } else if (l.left < r.left) {
        row.scrollLeft -= r.left - l.left;
      }
    }
    if (row) {
      var here = row.querySelector('a[aria-current="page"]');
      if (here) { reveal(here); }
      row.addEventListener("focusin", function (e) {
        var link = e.target && e.target.closest ? e.target.closest("a") : null;
        if (link) { reveal(link); }
      });
    }
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); } else { start(); }
})();
