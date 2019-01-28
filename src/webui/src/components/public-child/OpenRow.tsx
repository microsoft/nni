import * as React from 'react';
import PaiTrialLog from '../public-child/PaiTrialLog';
import TrialLog from '../public-child/TrialLog';
import JSONTree from 'react-json-tree';
import { TableObj } from '../../static/interface';
import { Row, Tabs } from 'antd';
const TabPane = Tabs.TabPane;

interface OpenRowProps {
    trainingPlatform: string;
    showLogModalOverview: Function;
    record: TableObj;
}

class OpenRow extends React.Component<OpenRowProps, {}> {

    constructor(props: OpenRowProps) {
        super(props);

    }

    render() {
        const { trainingPlatform, record, showLogModalOverview } = this.props;

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
                                    <JSONTree
                                        hideRoot={true}
                                        shouldExpandNode={() => true}  // default expandNode
                                        getItemString={() => (<span />)}  // remove the {} items
                                        data={openRowDataSource}
                                    />
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
                                        showLogModal={showLogModalOverview}
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