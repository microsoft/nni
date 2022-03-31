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


// Hide navigation bar when it's too short
// Hide TOC header when it coincides with page title
function hide_nav() {
    const d = $('nav.md-tabs[data-md-component="tabs"]');
    if (d.find('li').length <= 1) {
        d.addClass('hidden');
    }
}

function hide_toc_header() {
    const d = $(".md-nav__title.md-nav__title--site");
    // https://stackoverflow.com/questions/11362085/jquery-get-text-for-element-without-children-text
    const pageTitle = $("#index--page-root").clone().children().remove().end().text();
    if (d.text().trim() == pageTitle) {
        d.hide();
    }
}

$(document).ready(function() {
    hide_nav();
    hide_toc_header();
});
