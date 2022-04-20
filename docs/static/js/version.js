// Uncomment the following for debug
// READTHEDOCS_DATA = {
//     "ad_free": false,
//     "api_host": "https://readthedocs.org",
//     "build_date": "2022-01-25T06:27:55Z",
//     "builder": "sphinx",
//     "canonical_url": null,
//     "commit": "ca66e346",
//     "docroot": "/docs/en_US/",
//     "features": { "docsearch_disabled": false },
//     "global_analytics_code": "UA-17997319-1",
//     "language": "en",
//     "page": "Tutorial",
//     "programming_language": "words",
//     "project": "nni",
//     "proxied_api_host": "/_",
//     "source_suffix": ".rst",
//     "subprojects": { "nni-zh": "https://nni.readthedocs.io/zh/stable/" },
//     "theme": "sphinx_material",
//     "user_analytics_code": "UA-136029994-1",
//     "version": "latest"
// };

// READTHEDOCS_VERSIONS = [
//     ["latest", "/en/latest/"],
//     ["stable", "/en/stable/"],
//     ["v2.6", "/en/v2.6/"],
//     ["v2.5", "/en/v2.5/"],
//     ["v2.4", "/en/v2.4/"],
//     ["v2.3", "/en/v2.3/"]
// ];
// The above code is injected by readthedocs in production.


function create_dropdown(button_text, items) {
    const dropdown = document.createElement("div");
    dropdown.className = "md-flex__cell md-flex__cell--shrink drop";
    const button = document.createElement("button");
    button.innerHTML = button_text;
    const content = document.createElement("ul");
    // content.className = "dropdown-content md-hero";
    dropdown.appendChild(button);
    dropdown.appendChild(content);

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
        $(dropdown).find("ul").addClass("active");
        $(dropdown).find("button").addClass("active");
        e.stopPropagation();
    })
    $(document).click(function () {
        $(".drop").find(".active").removeClass("active");
    })
    return dropdown;
}

function remove_version_dropdown() {
    $(".navheader").children().last().remove();
}

function add_version_dropdown() {
    const prev_versions = Object.assign({}, ...READTHEDOCS_VERSIONS.map(([k, v]) => ({ [k]: v })));

    const current_version = 'v: ' + READTHEDOCS_DATA["version"];
    $(".navheader").append(create_dropdown(current_version, prev_versions));
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

    $(".navheader").append(create_dropdown(language_dropdown[current_language], {
        [language_dropdown['en']]: get_dropdown_href('en'),
        [language_dropdown['zh']]: get_dropdown_href('zh')
    }))
}

$(document).ready(function () {
    remove_version_dropdown();
    add_language_dropdown();
    add_version_dropdown();
});
