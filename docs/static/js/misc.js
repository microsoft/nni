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

// Expand link
function expand_link() {
    // on load, collapse all links without active on the inside
    $(".md-nav__expand").filter(function (index) {
        return $(".md-nav__link--active", this).length >= 1;
    }).addClass("md-nav__expand--active");

    // bind click events
    $(".md-nav__expand > a").click(function (e) {
        if (window.matchMedia("only screen and (min-width: 76.2em)").matches) {
            // not a drawer
            e.preventDefault();
            
            const target = $(e.target).parent();
            if (target.hasClass("md-nav__expand--active")) {
                target.removeClass("md-nav__expand--active");
            } else {
                target.addClass("md-nav__expand--active");
            }

            return false;
        }
        return true;
    });
}

$(document).ready(function() {
    hide_nav();
    expand_link();
});
