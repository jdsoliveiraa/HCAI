/* Variant viewer: show one language x identity at a time. Without JS every variant is listed. */
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    var controls = document.querySelector(".variant-controls");
    var sections = document.querySelectorAll("section.variant");
    if (!controls || !sections.length) { return; }
    controls.hidden = false;

    function value(name) {
      var checked = controls.querySelector('input[name="' + name + '"]:checked');
      return checked ? checked.value : "";
    }
    function show() {
      var lang = value("v-lang");
      var identity = value("v-identity");
      for (var i = 0; i < sections.length; i++) {
        var s = sections[i];
        s.hidden = !(s.getAttribute("data-lang") === lang && s.getAttribute("data-identity") === identity);
      }
    }
    controls.addEventListener("change", show);
    show();
  });
})();
