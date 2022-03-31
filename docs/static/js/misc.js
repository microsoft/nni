// Fix the hero text effect on large screens
function resetHeroHidden() {
    const scrollAmount = window.scrollY;
    if (window.matchMedia("only screen and (min-width: 76.25em)").matches) {
        // only enable this on large screens
        if (scrollAmount == 0) {
            $(".md-hero").attr("data-md-state", "");
        } else {
            $(".md-hero").attr("data-md-state", "hidden");
        }
    }
}

// https://github.com/bashtage/sphinx-material/blob/6e0ef822e58df57d6a9de5a58dc40c17fc34f557/sphinx_material/sphinx_material/static/javascripts/application.js#L1384
$(window).on("scroll", resetHeroHidden);
$(window).on("resize", resetHeroHidden);
$(window).on("orientationchange", resetHeroHidden);


// Sidebar header
$(document).ready(function() {
    let language = "en";  // default english
    try {
        language = READTHEDOCS_DATA["language"];
    } catch (e) {}

    const title = $(".md-sidebar--secondary .md-nav--secondary label.md-nav__title");
    if (language == "en") {
        title.text("On this page");
    } else if (language == "zh") {
        title.text("本页内容");
    }
});
