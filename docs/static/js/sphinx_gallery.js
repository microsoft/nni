$(document).ready(function() {
    const downloadNote = $(".sphx-glr-download-link-note.admonition.note");
    if (downloadNote.length > 0) {
        const githubLink = "https://github.com/microsoft/nni/blob/" + GIT_COMMIT_ID + "/examples/" + PAGENAME + ".py";
        const notebookLink = $(".sphx-glr-download-jupyter .reference.download").attr("href");

        // link to generated notebook file
        const colabLink = "https://colab.research.google.com/github/microsoft/nni/blob/" + GIT_COMMIT_ID +
            "/docs/source/" + PAGENAME + ".ipynb";

        downloadNote.removeClass("admonition");
        // the image links are stored in layout.html
        // to leverage jinja engine
        downloadNote.html(`
            <a class="notebook-action-link" href="${notebookLink}">
                <div class="notebook-action-div">
                    <img src="${GALLERY_LINKS.notebook}"/>
                    <div>Download Notebook</div>
                </div>
            </a>
            <a class="notebook-action-link" href="${colabLink}">
                <div class="notebook-action-div">
                    <img src="${GALLERY_LINKS.colab}"/>
                    <div>Run in Google Colab</div>
                </div>
            </a>
            <a class="notebook-action-link" href="${githubLink}">
                <div class="notebook-action-div">
                    <img src="${GALLERY_LINKS.github}"/>
                    <div>View on GitHub</div>
                </div>
            </a>
        `);
    }
});
