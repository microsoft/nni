import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import { TableObj } from '../../static/interface';
import { Row, Tabs, Button, message, Modal } from 'antd';
import { MANAGER_IP } from '../../static/const';
import '../../static/style/overview.scss';
import '../../static/style/copyParameter.scss';
import JSONTree from 'react-json-tree';
const TabPane = Tabs.TabPane;

interface OpenRowProps {
    trainingPlatform: string;
    record: TableObj;
    logCollection: boolean;
    multiphase: boolean;
}

interface OpenRowState {
    isShowFormatModal: boolean;
    formatStr: string;
}

class OpenRow extends React.Component<OpenRowProps, OpenRowState> {

    public _isMounted: boolean;
    constructor(props: OpenRowProps) {
        super(props);
        this.state = {
            isShowFormatModal: false,
            formatStr: ''
        };
    }

    showFormatModal = (record: TableObj) => {
        // get copy parameters
        const params = JSON.stringify(record.description.parameters, null, 4);
        // open modal with format string
        if (this._isMounted === true) {
            this.setState(() => ({ isShowFormatModal: true, formatStr: params }));
        }
    }

    hideFormatModal = () => {
        // close modal, destroy state format string data
        if (this._isMounted === true) {
            this.setState(() => ({ isShowFormatModal: false, formatStr: '' }));
        }
    }

    copyParams = () => {
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

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }
    render() {
        const { trainingPlatform, record, logCollection, multiphase } = this.props;
        const { isShowFormatModal, formatStr } = this.state;
        let isClick = false;
        let isHasParameters = true;
        if (record.description.parameters.error) {
            isHasParameters = false;
        }
        const openRowDataSource = record.description.parameters;
        const trialink: string = `${MANAGER_IP}/trial-jobs/${record.id}`;
        const logPathRow = record.description.logPath !== undefined
            ?
            record.description.logPath
            :
            'This trial\'s log path are not available.';
        return (
            <Row className="openRowContent hyperpar">
                <Tabs tabPosition="left" className="card">
                    <TabPane tab="Parameters" key="1">
                        {
                            multiphase
                                ?
                                <Row className="link">
                                    Trails for multiphase experiment will return a set of parameters,
                                    we are listing the latest parameter in webportal.
                                    <br />
                                    For the entire parameter set, please refer to the following "
                                    <a href={trialink} target="_blank">{trialink}</a>".
                                </Row>
                                :
                                <div />
                        }
                        {
                            isHasParameters
                                ?
                                <Row id="description">
                                    <Row className="bgHyper">
                                        {
                                            isClick
                                                ?
                                                <pre>{JSON.stringify(openRowDataSource, null, 4)}</pre>
                                                :
                                                <JSONTree
                                                    hideRoot={true}
                                                    shouldExpandNode={() => true}  // default expandNode
                                                    getItemString={() => (<span />)}  // remove the {} items
                                                    data={openRowDataSource}
                                                />
                                        }
                                    </Row>
                                    <Row className="copy">
                                        <Button
                                            onClick={this.showFormatModal.bind(this, record)}
                                        >
                                            Copy as json
                                        </Button>
                                    </Row>
                                </Row>
                                :
                                <Row className="logpath">
                                    <span className="logName">Error: </span>
                                    <span className="error">'This trial's parameters are not available.'</span>
                                </Row>
                        }
                    </TabPane>
                    <TabPane tab="Log" key="2">
                        {
                            trainingPlatform !== 'local'
                                ?
                                <PaiTrialLog
                                    logStr={logPathRow}
                                    id={record.id}
                                    logCollection={logCollection}
                                />
                                :
                                <TrialLog logStr={logPathRow} id={record.id} />
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
