import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import { TableObj } from '../../static/interface';
import { Row, Tabs, Button, message } from 'antd';
import { MANAGER_IP } from '../../static/const';
import '../../static/style/overview.scss';
import JSONTree from 'react-json-tree';
const TabPane = Tabs.TabPane;

interface OpenRowProps {
    trainingPlatform: string;
    record: TableObj;
    logCollection: boolean;
    multiphase: boolean;
}

interface OpenRowState {
    idList: Array<string>;
}

class OpenRow extends React.Component<OpenRowProps, OpenRowState> {

    constructor(props: OpenRowProps) {
        super(props);
        this.state = {
            idList: ['']
        };

    }

    copyParams = (record: TableObj) => {
        // json format
        const params = JSON.stringify(record.description.parameters, null, 4);
        if (copy(params)) {
            message.destroy();
            message.success('Success copy parameters to clipboard in form of python dict !', 3);
            const { idList } = this.state;
            const copyIdList: Array<string> = idList;
            copyIdList[copyIdList.length - 1] = record.id;
            this.setState(() => ({
                idList: copyIdList
            }));
        } else {
            message.destroy();
            message.error('Failed !', 2);
        }
    }

    render() {
        const { trainingPlatform, record, logCollection, multiphase } = this.props;
        const { idList } = this.state;
        let isClick = false;
        let isHasParameters = true;
        if (idList.indexOf(record.id) !== -1) { isClick = true; }
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
                                            onClick={this.copyParams.bind(this, record)}
                                        >
                                            Copy as python
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
            </Row >
        );
    }
}

export default OpenRow;
