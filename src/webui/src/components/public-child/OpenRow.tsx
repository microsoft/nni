import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import MessageInfo from '../Modal/MessageInfo';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { Trial } from '../../static/model/trial';
import { Stack, PrimaryButton, Pivot, PivotItem, Dialog, DialogFooter, DefaultButton } from 'office-ui-fabric-react';
import { MANAGER_IP } from '../../static/const';
import '../../static/style/overview.scss';
import '../../static/style/copyParameter.scss';
import JSONTree from 'react-json-tree';

interface OpenRowProps {
    trialId: string;
}

interface OpenRowState {
    isShowFormatModal: boolean;
    formatStr: string;
    typeInfo: string;
    info: string;
    isHidenInfo: boolean;
}

class OpenRow extends React.Component<OpenRowProps, OpenRowState> {

    constructor(props: OpenRowProps) {
        super(props);
        this.state = {
            isShowFormatModal: false,
            formatStr: '',
            typeInfo: '',
            info: '',
            isHidenInfo: true
        };
    }

    showFormatModal = (trial: Trial): void => {
        // get copy parameters
        const params = JSON.stringify(trial.description.parameters, null, 4);
        // open modal with format string
        this.setState({ isShowFormatModal: true, formatStr: params });
    }

    hideFormatModal = (): void => {
        // close modal, destroy state format string data
        this.setState({ isShowFormatModal: false, formatStr: '' });
    }

    hideMessageInfo = (): void => {
        this.setState(() => ({ isHidenInfo: true }));
    }
    /**
     * info: message content
     * typeInfo: message type: success | error...
     * continuousTime: show time, 2000ms 
     */
    getCopyStatus = (info: string, typeInfo: string): void => {
        this.setState(() => ({ info, typeInfo, isHidenInfo: false }));
        setTimeout(this.hideMessageInfo, 2000);
    }

    copyParams = (): void => {
        // json format
        const { formatStr } = this.state;
        if (copy.default(formatStr)) {
            this.getCopyStatus('Success copy parameters to clipboard in form of python dict !', 'success');
        } else {
            this.getCopyStatus('Failed !', 'error');
        }
        this.hideFormatModal();
    }

    render(): React.ReactNode {
        const { isShowFormatModal, formatStr, isHidenInfo, typeInfo, info } = this.state;
        const trialId = this.props.trialId;
        const trial = TRIALS.getTrial(trialId);
        const trialLink: string = `${MANAGER_IP}/trial-jobs/${trialId}`;
        const logPathRow = trial.info.logPath || 'This trial\'s log path is not available.';
        const multiProgress = trial.info.hyperParameters === undefined ? 0 : trial.info.hyperParameters.length;
        return (
            <Stack className="openRowContent hyperpar">
                <Pivot>
                    <PivotItem headerText="Parameters" key="1" itemIcon="Recent">
                        {
                            EXPERIMENT.multiPhase
                                ?
                                <Stack className="link">
                                    {
                                        `
                                        Trails for multiphase experiment will return a set of parameters,
                                        we are listing the latest parameter in webportal.
                                        For the entire parameter set, please refer to the following "
                                        `
                                    }
                                    <a href={trialLink} rel="noopener noreferrer" target="_blank">{trialLink}</a>{`".`}
                                    <div>Current Phase: {multiProgress}.</div>
                                </Stack>
                                :
                                null
                        }
                        {
                            trial.info.hyperParameters !== undefined
                                ?
                                <Stack id="description">
                                    <Stack className="bgHyper">
                                        <JSONTree
                                            hideRoot={true}
                                            shouldExpandNode={(): boolean => true}  // default expandNode
                                            // getItemString={(): null => (<span />)}  // remove the {} items
                                            getItemString={(): null => null}  // remove the {} items
                                            data={trial.description.parameters}
                                        />
                                    </Stack>
                                    <Stack className="copy" styles={{ root: { width: 128 } }}>
                                        <PrimaryButton
                                            onClick={this.showFormatModal.bind(this, trial)}
                                            text="Copy as json"
                                        />
                                    </Stack>
                                </Stack>
                                :
                                <Stack className="logpath">
                                    <span className="logName">Error: </span>
                                    <span className="error">{`This trial's parameters are not available.'`}</span>
                                </Stack>
                        }
                    </PivotItem>
                    <PivotItem headerText="Log" key="2">
                        {
                            // FIXME: this should not be handled in web UI side
                            EXPERIMENT.trainingServicePlatform !== 'local'
                                ?
                                <PaiTrialLog
                                    logStr={logPathRow}
                                    id={trialId}
                                    logCollection={EXPERIMENT.logCollectionEnabled}
                                />
                                :
                                <TrialLog logStr={logPathRow} id={trialId} />
                        }
                    </PivotItem>
                </Pivot>
                {
                    isShowFormatModal && <Dialog
                        hidden={false}
                        onDismiss={this.hideFormatModal}
                        className="format"
                        minWidth={600}
                        dialogContentProps={{
                            // type: DialogType.normal,
                            title: 'Format',
                            closeButtonAriaLabel: 'Close',
                            // subText: 'Do you want to send this message without a subject?'
                        }}
                    >
                        {/* write string in pre to show format string */}
                        <pre className="formatStr">{formatStr}</pre>
                        <DialogFooter>
                            <PrimaryButton onClick={this.copyParams} text="Copy" />
                            <DefaultButton onClick={this.hideFormatModal} text="Cancel" />
                        </DialogFooter>
                    </Dialog>
                }
                {/* copy success | failed message info */}
                {!isHidenInfo && <MessageInfo typeInfo={typeInfo} info={info} />}
            </Stack >
        );
    }
}

export default OpenRow;
