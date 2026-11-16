/* Interface language switch (EN / PT-PT). Only interface strings ([data-i18n]) change; repository prose stays
   English. The choice is remembered in localStorage when the browser allows it. */
(function () {
  "use strict";
  var STRINGS = /*STRINGS*/null;
  var KEY = "hcai-lang";

  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(lang) {
    try { window.localStorage.setItem(KEY, lang); } catch (e) { /* not saved: the page still switches */ }
  }

  var current = stored() === "pt" ? "pt" : "en";

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
    var button = document.getElementById("lang-toggle");
    if (button) {
      var other = lang === "pt" ? "en" : "pt";
      button.textContent = t("lang.switch", lang);
      button.setAttribute("lang", other);
      button.setAttribute("aria-label", t("lang.switch.label", lang));
    }
    document.dispatchEvent(new CustomEvent("hcai:lang", { detail: lang }));
  }

  window.hcaiT = function (key) { return t(key); };

  document.addEventListener("DOMContentLoaded", function () {
    if (current !== "en") { apply(current); }
    var button = document.getElementById("lang-toggle");
    if (button) {
      button.addEventListener("click", function () {
        var next = current === "pt" ? "en" : "pt";
        store(next);
        apply(next);
      });
    }
  });
})();
