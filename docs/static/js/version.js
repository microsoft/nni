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
    var dropdown = document.createElement("div");
    dropdown.className = "md-flex__cell md-flex__cell--shrink dropdown";
    var button = document.createElement("button");
    button.className = "dropdownbutton";
    button.innerHTML = button_text;
    var content = document.createElement("div");
    content.className = "dropdown-content md-hero";
    dropdown.appendChild(button);
    dropdown.appendChild(content);

    for (var key in items) {
        if (items.hasOwnProperty(key)) {
            console.log(key, items[key]);
            var a = document.createElement("a");
            a.innerHTML = key;
            a.title = key;
            a.href = items[key];
            content.appendChild(a);
        }
    }

    return dropdown;
}

function remove_version_dropdown() {
    $(".navheader").children().last().remove();
}

function add_version_dropdown() {
    var prev_versions = Object.assign({}, ...READTHEDOCS_VERSIONS.map(([k, v]) => ({ [k]: v })));
    console.log(prev_versions);

    var current_version = 'v: ' + READTHEDOCS_DATA["version"];
    $(".navheader").append(create_dropdown(current_version, prev_versions));
}

function add_language_dropdown() {
    var language_dropdown = {
        'en': 'English',
        'zh': '简体中文'
    };
    var current_language = 'en';
    var pathname_prefix = window.location.pathname.split('/');
    if (pathname_prefix.length > 1 && language_dropdown.hasOwnProperty(pathname_prefix[1])) {
        current_language = pathname_prefix[1];
    }

    function get_dropdown_href(lang) {
        var pathname = window.location.pathname.split('/');
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
