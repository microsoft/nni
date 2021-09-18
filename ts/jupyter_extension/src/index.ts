import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, IFrame } from '@jupyterlab/apputils';
import { PageConfig } from '@jupyterlab/coreutils';
import { ILauncher } from '@jupyterlab/launcher';
import { LabIcon } from '@jupyterlab/ui-components';

const nniIconSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 162 84">
  <polygon fill="#0b6bac" points="0,84 12,84 34,18 56,84 68,84 87,27 75,27 62,66 40,0 28,0 0,84"/>
  <polygon fill="#0b6bac" points="94,84 106,84 125,27 113,27 100,66 90,36 84,54 94,84"/>
  <polygon fill="#0b6bac" points="122,0 128,18 140,18 134,0 122,0"/>
  <polygon fill="#0b6bac" points="131,27 150,84 162,84 143,27 131,27"/>
</svg>
`;
const nniIcon = new LabIcon({ name: 'nni', svgstr: nniIconSvg });

class NniWidget extends IFrame {
    constructor() {
        super({
            sandbox: [
                'allow-same-origin',
                'allow-scripts',
            ]
        });
        this.url = PageConfig.getBaseUrl() + 'nni/index';
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
            shell.add(new NniWidget(), 'main');
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
