import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, IFrame, Dialog, showDialog } from '@jupyterlab/apputils';
import { PageConfig } from '@jupyterlab/coreutils';
import { ILauncher } from '@jupyterlab/launcher';
import { LabIcon } from '@jupyterlab/ui-components';
import React from 'react';

const nniIconSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 162 84">
  <polygon fill="#0b6bac" points="0,84 12,84 34,18 56,84 68,84 87,27 75,27 62,66 40,0 28,0 0,84"/>
  <polygon fill="#0b6bac" points="94,84 106,84 125,27 113,27 100,66 90,36 84,54 94,84"/>
  <polygon fill="#0b6bac" points="122,0 128,18 140,18 134,0 122,0"/>
  <polygon fill="#0b6bac" points="131,27 150,84 162,84 143,27 131,27"/>
</svg>
`;
const nniIcon = new LabIcon({ name: 'nni', svgstr: nniIconSvg });

const NNI_URL = PageConfig.getBaseUrl() + 'nni/index';
class NniWidget extends IFrame {
    constructor(url) {
        super({
            sandbox: [
                'allow-same-origin',
                'allow-scripts',
            ]
        });
        this.url = url;
        this.id = 'nni';
        this.title.label = 'NNI';
        this.title.icon = nniIcon;
        this.title.closable = true;
    }
}

async function activate(app: JupyterFrontEnd, palette: ICommandPalette, launcher: ILauncher | null) {
    console.log('nni extension is activated');
    const { commands, shell } = app;
    const command = 'nni';
    const category = 'Other';

    commands.addCommand(command, {
        label: 'NNI',
        caption: 'NNI',
        icon: (args) => (args.isPalette ? null : nniIcon),
        execute: () => {
            fetch(NNI_URL).then(async (resp) => {
                if (resp.status !== 200) {
                    showDialog({
                        title: 'NNI-HPO Launcher Error',
                        body: React.createElement("div", null,
                            "please run command:",
                            React.createElement("div", { style: { color: 'blue', fontSize: "14px", lineHeight: "28px" } }, "nnictl create --config experiment.yml")),
                        buttons: [Dialog.warnButton({ label: 'OK' })]
                    });
                    return;
                }
                shell.add(new NniWidget(NNI_URL), 'main');
            });
        }
    });

    palette.addItem({ command, category });

    if (launcher) {
        launcher.add({ command, category });
    }
}

const extension: JupyterFrontEndPlugin<void> = {
    id: 'nni',
    autoStart: true,
    optional: [ILauncher],
    requires: [ICommandPalette],
    activate,
};

export default extension;
