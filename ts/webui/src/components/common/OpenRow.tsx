import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import { Stack, PrimaryButton, Pivot, PivotItem, DefaultButton } from '@fluentui/react';
import JSONTree from 'react-json-tree';
import { Trial } from '@model/trial';
import { MANAGER_IP, RETIARIIPARAMETERS } from '@static/const';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { reformatRetiariiParameter } from '@static/function';
import PaiTrialLog from './PaiTrialLog';
import TrialLog from './TrialLog';
import MessageInfo from './MessageInfo';
import PanelMonacoEditor from './PanelMonacoEditor';
import '@style/experiment/overview/overview.scss';
import '@style/openRow.scss';

/**
 *  netron URL must be synchronized with ts/nni_manager/rest_server/index.ts`.
 *  Remember to update it if the value is changed or this file is moved.
 **/

interface OpenRowProps {
    trialId: string;
}

interface OpenRowState {
    typeInfo: string;
    info: string;
    isHidenInfo: boolean;
    showRetiaParamPanel: boolean;
}

class OpenRow extends React.Component<OpenRowProps, OpenRowState> {
    constructor(props: OpenRowProps) {
        super(props);
        this.state = {
            typeInfo: '',
            info: '',
            isHidenInfo: true,
            showRetiaParamPanel: false
        };
    }

    hideMessageInfo = (): void => {
        this.setState(() => ({ isHidenInfo: true }));
    };

    hideRetiaParam = (): void => {
        this.setState(() => ({ showRetiaParamPanel: false }));
    };

    isshowRetiaParamPanel = (): void => {
        this.setState(() => ({ showRetiaParamPanel: true }));
    };

    /**
     * info: message content
     * typeInfo: message type: success | error...
     * continuousTime: show time, 2000ms
     */
    getCopyStatus = (info: string, typeInfo: string): void => {
        this.setState(() => ({ info, typeInfo, isHidenInfo: false }));
        setTimeout(this.hideMessageInfo, 2000);
    };

    copyParams = (trial: Trial): void => {
        // get copy parameters
        const params = JSON.stringify(reformatRetiariiParameter(trial.description.parameters as any), null, 4);
        if (copy.default(params)) {
            this.getCopyStatus('Success copy parameters to clipboard in form of python dict !', 'success');
        } else {
            this.getCopyStatus('Failed !', 'error');
        }
    };

    openTrialLog = (filename: string): void => {
        window.open(`${MANAGER_IP}/trial-file/${this.props.trialId}/${filename}`);
    };

    openModelOnnx = (): void => {
        // TODO: netron might need prefix.
        window.open(`/netron/index.html?url=${MANAGER_IP}/trial-file/${this.props.trialId}/model.onnx`);
    };

    render(): React.ReactNode {
        const { isHidenInfo, typeInfo, info, showRetiaParamPanel } = this.state;
        const trialId = this.props.trialId;
        const trial = TRIALS.getTrial(trialId);
        const logPathRow = trial.info.logPath || "This trial's log path is not available.";
        const originParameters = trial.description.parameters;
        const hasVisualHyperParams = RETIARIIPARAMETERS in originParameters;
        return (
            <Stack className='openRow'>
                <Stack className='openRowContent'>
                    <Pivot>
                        <PivotItem headerText='Parameters' key='1' itemIcon='TestParameter'>
                            {trial.info.hyperParameters !== undefined ? (
                                <Stack id='description'>
                                    <Stack className='bgHyper'>
                                        <JSONTree
                                            hideRoot={true}
                                            shouldExpandNode={(): boolean => true} // default expandNode
                                            getItemString={(): null => null} // remove the {} items
                                            data={reformatRetiariiParameter(originParameters as any)}
                                        />
                                    </Stack>
                                    <Stack horizontal className='copy'>
                                        <PrimaryButton
                                            onClick={this.copyParams.bind(this, trial)}
                                            text='Copy as json'
                                            styles={{ root: { width: 128, marginRight: 10 } }}
                                        />
                                        {hasVisualHyperParams && (
                                            <DefaultButton
                                                onClick={this.isshowRetiaParamPanel}
                                                text='Original parameters'
                                            />
                                        )}
                                        {/* copy success | failed message info */}
                                        {!isHidenInfo && <MessageInfo typeInfo={typeInfo} info={info} />}
                                        {showRetiaParamPanel && (
                                            <PanelMonacoEditor
                                                hideConfigPanel={this.hideRetiaParam}
                                                panelName='Retiarii parameters'
                                                panelContent={JSON.stringify(originParameters, null, 2)}
                                            />
                                        )}
                                    </Stack>
                                </Stack>
                            ) : (
                                <Stack className='logpath'>
                                    <span className='logName'>Error: </span>
                                    <span className='error'>{`This trial's parameters are not available.'`}</span>
                                </Stack>
                            )}
                        </PivotItem>
                        <PivotItem headerText='Log' key='2' itemIcon='M365InvoicingLogo'>
                            {
                                // FIXME: this should not be handled in web UI side
                                EXPERIMENT.trainingServicePlatform !== 'local' ? (
                                    <PaiTrialLog
                                        logStr={logPathRow}
                                        id={trialId}
                                        logCollection={EXPERIMENT.logCollectionEnabled}
                                    />
                                ) : (
                                    <div>
                                        <TrialLog logStr={logPathRow} id={trialId} />
                                        {/* view each trial log in drawer*/}
                                        <div id='trialog'>
                                            <div className='copy' style={{ marginTop: 15 }}>
                                                <PrimaryButton
                                                    onClick={this.openTrialLog.bind(this, 'trial.log')}
                                                    text='View trial log'
                                                />
                                                <PrimaryButton
                                                    onClick={this.openTrialLog.bind(this, 'stderr')}
                                                    text='View trial error'
                                                    styles={{ root: { marginLeft: 15 } }}
                                                />
                                                <PrimaryButton
                                                    onClick={this.openTrialLog.bind(this, 'stdout')}
                                                    text='View trial stdout'
                                                    styles={{ root: { marginLeft: 15 } }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                )
                            }
                        </PivotItem>
                        {EXPERIMENT.metadata.tag.includes('retiarii') ? (
                            <PivotItem headerText='Visualization' key='3' itemIcon='FlowChart'>
                                <div id='visualization'>
                                    <div id='visualizationText'>Visualize models with 3rd-party tools.</div>
                                    <PrimaryButton
                                        onClick={this.openModelOnnx.bind(this)}
                                        text='Netron'
                                        styles={{ root: { marginLeft: 15 } }}
                                    />
                                </div>
                            </PivotItem>
                        ) : null}
                    </Pivot>
                </Stack>
            </Stack>
        );
    }
}

export default OpenRow;
