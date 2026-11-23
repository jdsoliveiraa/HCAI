/* Variant viewer: show one language x identity at a time, or two identities side by side ("Compare with").
   Without JS every variant is listed. */
(function () {
  "use strict";

  /* Which variants to show: the chosen identity first, then the one it is compared with (if another). */
  function shown(lang, identity, compare) {
    var ids = [identity];
    if (compare && compare !== identity) { ids.push(compare); }
    return { lang: lang, identities: ids };
  }

  if (typeof module !== "undefined" && module.exports) { module.exports = { shown: shown }; }
  if (typeof document === "undefined") { return; }

  function start() {
    var controls = document.querySelector(".variant-controls");
    var holder = document.querySelector(".variants");
    var sections = document.querySelectorAll("section.variant");
    if (!controls || !holder || !sections.length) { return; }
    var compare = document.getElementById("v-compare");
    controls.hidden = false;

    function value(name) {
      var checked = controls.querySelector('input[name="' + name + '"]:checked');
      return checked ? checked.value : "";
    }
    function show() {
      var want = shown(value("v-lang"), value("v-identity"), compare ? compare.value : "");
      for (var i = 0; i < sections.length; i++) {
        var s = sections[i];
        var at = want.identities.indexOf(s.getAttribute("data-identity"));
        s.hidden = !(s.getAttribute("data-lang") === want.lang && at >= 0);
        s.style.order = at > 0 ? "1" : "";  // the chosen identity on the left, the compared one on the right
      }
      holder.classList.toggle("compare", want.identities.length > 1);
    }
    controls.addEventListener("change", show);
    show();
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); } else { start(); }
})();
