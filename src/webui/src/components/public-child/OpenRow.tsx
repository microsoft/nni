import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { Trial } from '../../static/model/trial';
import { Row, Tabs, Button, message, Modal } from 'antd';
import { MANAGER_IP } from '../../static/const';
import '../../static/style/overview.scss';
import '../../static/style/copyParameter.scss';
import JSONTree from 'react-json-tree';
const TabPane = Tabs.TabPane;

interface OpenRowProps {
    trialId: string;
}

interface OpenRowState {
    isShowFormatModal: boolean;
    formatStr: string;
}

class OpenRow extends React.Component<OpenRowProps, OpenRowState> {

    constructor(props: OpenRowProps) {
        super(props);
        this.state = {
            isShowFormatModal: false,
            formatStr: ''
        };
    }

    showFormatModal = (trial: Trial): void => {
        // get copy parameters
        const params = JSON.stringify(trial.info.hyperParameters, null, 4);
        // open modal with format string
        this.setState({ isShowFormatModal: true, formatStr: params });
    }

    hideFormatModal = (): void => {
        // close modal, destroy state format string data
        this.setState({ isShowFormatModal: false, formatStr: '' });
    }

    copyParams = (): void => {
        // json format
        const { formatStr } = this.state;
        if (copy(formatStr)) {
            message.destroy();
            message.success('Success copy parameters to clipboard in form of python dict !', 3);
        } else {
            message.destroy();
            message.error('Failed !', 2);
        }
        this.hideFormatModal();
    }

    render(): React.ReactNode {
        const { isShowFormatModal, formatStr } = this.state;
        const trialId = this.props.trialId;
        const trial = TRIALS.getTrial(trialId);
        const trialLink: string = `${MANAGER_IP}/trial-jobs/${trialId}`;
        const logPathRow = trial.info.logPath || 'This trial\'s log path is not available.';
        const multiProgress = trial.info.hyperParameters === undefined ? 0 : trial.info.hyperParameters.length;
        return (
            <Row className="openRowContent hyperpar">
                <Tabs tabPosition="left" className="card">
                    <TabPane tab="Parameters" key="1">
                        {
                            EXPERIMENT.multiPhase
                                ?
                                <Row className="link">
                                    Trails for multiphase experiment will return a set of parameters,
                                    we are listing the latest parameter in webportal.
                                    <br />
                                    For the entire parameter set, please refer to the following &quot;
                                    <a
                                        href={trialLink}
                                        rel="noopener noreferrer"
                                        target="_blank"
                                        style={{marginLeft: 2}}
                                    >
                                        {trialLink}
                                    </a>&quot;
                                    <br />
                                    Current Phase:{multiProgress}.
                                </Row>
                                :
                                <div />
                        }
                        {
                            trial.info.hyperParameters !== undefined
                                ?
                                <Row id="description">
                                    <Row className="bgHyper">
                                        <JSONTree
                                            hideRoot={true}
                                            shouldExpandNode={(): boolean => true}  // default expandNode
                                            getItemString={(): any => (<span />)}  // remove the {} items
                                            data={trial.description.parameters}
                                        />
                                    </Row>
                                    <Row className="copy">
                                        <Button
                                            onClick={this.showFormatModal.bind(this, trial)}
                                        >
                                            Copy as json
                                        </Button>
                                    </Row>
                                </Row>
                                :
                                <Row className="logpath">
                                    <span className="logName" style={{marginRight: 2}}>Error:</span>
                                    <span className="error">&apos;This trial&apos;s parameters are not available.&apos;</span>
                                </Row>
                        }
                    </TabPane>
                    <TabPane tab="Log" key="2">
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
                    </TabPane>
                </Tabs>
                <Modal
                    title="Format"
                    okText="Copy"
                    centered={true}
                    visible={isShowFormatModal}
                    onCancel={this.hideFormatModal}
                    maskClosable={false} // click mongolian layer don't close modal
                    onOk={this.copyParams}
                    destroyOnClose={true}
                    width="60%"
                    className="format"
                >
                    {/* write string in pre to show format string */}
                    <pre className="formatStr">{formatStr}</pre>
                </Modal>
            </Row >
        );
    }
}

export default OpenRow;
