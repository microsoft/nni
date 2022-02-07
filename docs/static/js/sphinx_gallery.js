`<div class="pytorch-call-to-action-links">
<div id="tutorial-type">{{ pagename }}</div>

<div id="google-colab-link">
    <img class="call-to-action-img" src="_static/images/gallery-colab.svg"/>
    <div>Run in Google Colab</div>
    <div class="call-to-action-mobile-view">Colab</div>
</div>
<div id="download-notebook-link">
    <img class="call-to-action-notebook-img" src="_static/img/gallery-download.svg"/>
    <div>Download Notebook</div>
    <div class="call-to-action-mobile-view">Notebook</div>
</div>
<div id="github-view-link">
    <img class="call-to-action-img" src="_static/images/gallery-github.svg"/>
    <div>View on GitHub</div>
    <div class="call-to-action-mobile-view">GitHub</div>
</div>
</div>`;

$(document).ready(function() {
    const downloadNote = $(".sphx-glr-download-link-note.admonition.note");
    if (downloadNote.length > 0) {
        const githubLink = "https://github.com/microsoft/nni/blob/" + GIT_COMMIT_ID + "/examples/" + PAGENAME + ".py";
        const notebookLink = $(".sphx-glr-download-jupyter .reference.download").attr("href");

        // link to generated notebook file
        const colabLink = "https://colab.research.google.com/github/microsoft/nni/blob/" + GIT_COMMIT_ID +
            "/docs/source/" + PAGENAME + ".ipynb";

        downloadNote.removeClass("admonition");
        downloadNote.html(`
            <a class="notebook-action-link" href="${colabLink}">
                <div class="notebook-action-div">
                    <img src="/_static/img/gallery-colab.svg"/>
                    <div>Run in Google Colab</div>
                </div>
            </a>
            <a class="notebook-action-link" href="${notebookLink}">
                <div class="notebook-action-div">
                    <img src="/_static/img/gallery-download.svg"/>
                    <div>Download Notebook</div>
                </div>
            </a>
            <a class="notebook-action-link" href="${githubLink}">
                <div class="notebook-action-div">
                    <img src="/_static/img/gallery-github.svg"/>
                    <div>View on GitHub</div>
                </div>
            </a>
        `);
    }
});
