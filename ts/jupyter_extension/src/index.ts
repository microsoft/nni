import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, IFrame } from '@jupyterlab/apputils';
import { PageConfig } from '@jupyterlab/coreutils';
import { ILauncher } from '@jupyterlab/launcher';

class NniWidget extends IFrame {
    constructor() {
        super({
            sandbox: [
                'allow-same-origin',
                'allow-scripts',
            ]
        });
        this.url = PageConfig.getBaseUrl() + 'nni/oview';
        this.id = 'nni';
        this.title.label = 'NNI';
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
