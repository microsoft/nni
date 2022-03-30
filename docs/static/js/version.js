try {
    READTHEDOCS_DATA;
} catch (e) {
    console.log('READTHEDOCS_DATA is undefined. In debug mode.');

    // mock info
    READTHEDOCS_DATA = {
        "ad_free": false,
        "api_host": "https://readthedocs.org",
        "build_date": "2022-01-25T06:27:55Z",
        "builder": "sphinx",
        "canonical_url": null,
        "commit": "ca66e346",
        "docroot": "/docs/en_US/",
        "features": { "docsearch_disabled": false },
        "global_analytics_code": "UA-17997319-1",
        "language": "en",
        "page": "Tutorial",
        "programming_language": "words",
        "project": "nni",
        "proxied_api_host": "/_",
        "source_suffix": ".rst",
        "subprojects": { "nni-zh": "https://nni.readthedocs.io/zh/stable/" },
        "theme": "sphinx_material",
        "user_analytics_code": "UA-136029994-1",
        "version": "latest"
    };
    
    READTHEDOCS_VERSIONS = [
        ["latest", "/en/latest/"],
        ["stable", "/en/stable/"],
        ["v2.6", "/en/v2.6/"],
        ["v2.5", "/en/v2.5/"],
        ["v2.4", "/en/v2.4/"],
        ["v2.3", "/en/v2.3/"],
        ["test-version", "/en/test-version"]
    ];
    // The above code is injected by readthedocs in production.
}

function create_dropdown(selector, button_text, items) {
    const dropdown = $(selector);
    const button = document.createElement("button");
    button.innerHTML = button_text;
    const content = document.createElement("ul");
    // content.className = "dropdown-content md-hero";
    dropdown.append(button);
    dropdown.append(content);

    for (const key in items) {
        if (items.hasOwnProperty(key)) {
            const li = document.createElement("li");
            const a = document.createElement("a");
            a.className = "md-nav__link"
            a.innerHTML = key;
            a.title = key;
            a.href = items[key];
            li.appendChild(a);
            content.appendChild(li);
        }
    }

    $(button).click(function (e) {
        // first close all others.
        $(".drop").find(".active").removeClass("active");
        dropdown.find("ul").addClass("active");
        dropdown.find("button").addClass("active");
        e.stopPropagation();
    })
    $(document).click(function () {
        $(".drop").find(".active").removeClass("active");
    })
    return dropdown;
}

function add_version_dropdown() {
    const prev_versions = Object.assign(
        {},
        ...READTHEDOCS_VERSIONS
            .filter(([k, v]) => (k === 'stable' || k == 'latest' || k.startsWith('v')))
            .map(([k, v]) => ({ [k]: v }))
    );

    const current_version = 'v: ' + READTHEDOCS_DATA["version"];
    create_dropdown(".drop.version", current_version, prev_versions);
}

function add_language_dropdown() {
    const language_dropdown = {
        'en': 'English',
        'zh': '简体中文'
    };
    let current_language = 'en';
    const pathname_prefix = window.location.pathname.split('/');
    if (pathname_prefix.length > 1 && language_dropdown.hasOwnProperty(pathname_prefix[1])) {
        current_language = pathname_prefix[1];
    }

    function get_dropdown_href(lang) {
        let pathname = window.location.pathname.split('/');
        if (pathname.length > 1) {
            pathname[1] = lang;
        }
        return pathname.join('/');
    }

    create_dropdown(".drop.language", language_dropdown[current_language], {
        [language_dropdown['en']]: get_dropdown_href('en'),
        [language_dropdown['zh']]: get_dropdown_href('zh')
    });
}

function hide_nav() {
    const d = $('nav.md-tabs[data-md-component="tabs"]');
    if (d.find('li').length <= 1) {
        d.addClass('hidden');
    }
}

$(document).ready(function () {
    hide_nav();
    add_language_dropdown();
    add_version_dropdown();
});

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
