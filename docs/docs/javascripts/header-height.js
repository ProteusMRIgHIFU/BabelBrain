// BabelBrain: publish the header's current height as the CSS variable
// --bb-header-height, so sticky sidebars and anchor jumps sit below the
// header when the navigation tabs wrap onto two or more rows.
(function () {
  function watch() {
    var header = document.querySelector(".md-header");
    if (!header) return;
    var root = document.documentElement;
    var update = function () {
      root.style.setProperty("--bb-header-height", header.offsetHeight + "px");
    };
    update();
    if ("ResizeObserver" in window) {
      new ResizeObserver(update).observe(header);
    } else {
      window.addEventListener("resize", update);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watch);
  } else {
    watch();
  }
})();
