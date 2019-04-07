import * as React from 'react';
import * as copy from 'copy-to-clipboard';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import { TableObj } from '../../static/interface';
import { Row, Tabs, Button, message } from 'antd';
import JSONTree from 'react-json-tree';
const TabPane = Tabs.TabPane;

interface OpenRowProps {
    trainingPlatform: string;
    record: TableObj;
    logCollection: boolean;
}

class OpenRow extends React.Component<OpenRowProps, {}> {

    constructor(props: OpenRowProps) {
        super(props);

    }

    copyParams = (record: TableObj) => {
        let params = JSON.stringify(record.description.parameters);
        if (copy(params)) {
            message.success('Success copy parameters to clipboard in form of python dict !', 3);
        } else {
            message.error('Failed !', 2);
        }
    }

    render() {
        const { trainingPlatform, record, logCollection } = this.props;

        let isHasParameters = true;
        if (record.description.parameters.error) {
            isHasParameters = false;
        }
        const openRowDataSource = {
            parameters: record.description.parameters
        };
        const logPathRow = record.description.logPath !== undefined
            ?
            record.description.logPath
            :
            'This trial\'s logPath are not available.';
        return (
            <pre id="description" className="hyperpar">
                <Row className="openRowContent">
                    <Tabs tabPosition="left" className="card">
                        <TabPane tab="Parameters" key="1">
                            {
                                isHasParameters
                                    ?
                                    <div>
                                        <JSONTree
                                            hideRoot={true}
                                            shouldExpandNode={() => true}  // default expandNode
                                            getItemString={() => (<span />)}  // remove the {} items
                                            data={openRowDataSource.parameters}
                                        />
                                        <Button
                                            onClick={this.copyParams.bind(this, record)}
                                        >
                                            Copy as Python
                                        </Button>
                                    </div>
                                    :
                                    <div className="logpath">
                                        <span className="logName">Error: </span>
                                        <span className="error">'This trial's parameters are not available.'</span>
                                    </div>
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
                </Row>
            </pre>
        );
    }
}

export default OpenRow;