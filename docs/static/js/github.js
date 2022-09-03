function add_descriptive_text(texts) {
    const d = $('*[data-md-source="nni"]');
    // clear first
    d.find("ul").remove();

    let s = "";
    for (const text of texts) {
        s += '<li class="md-source__fact">' + text + '</li>';
    }
    d.find(".md-source__repository").append('<ul class="md-source__facts">' + s + '</ul>');
}

function kFormatter(num) {
    // https://stackoverflow.com/questions/9461621/format-a-number-as-2-5k-if-a-thousand-or-more-otherwise-900
    return Math.abs(num) > 999 ? Math.sign(num)*((Math.abs(num)/1000).toFixed(1)) + 'k' : Math.sign(num)*Math.abs(num);
}

$(document).ready(function() {
    add_descriptive_text(["View on GitHub"]);
    $.getJSON("https://api.github.com/repos/microsoft/nni", function (data) {
        add_descriptive_text([
            kFormatter(data["stargazers_count"]) + " stars",
            kFormatter(data["forks"]) + " forks",
        ]);
    });
});
