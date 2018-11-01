import * as React from 'react';
import JSONTree from 'react-json-tree';
import { Table } from 'antd';
import { TableObj } from '../../static/interface';
import { convertDuration } from '../../static/function';
import '../../static/style/tableStatus.css';

interface SuccessTableProps {
    tableSource: Array<TableObj>;
}

class SuccessTable extends React.Component<SuccessTableProps, {}> {

    constructor(props: SuccessTableProps) {
        super(props);

    }

    render() {
        const { tableSource } = this.props;

        let bgColor = '';
        const columns = [{
            title: 'Number',
            dataIndex: 'sequenceId',
            key: 'sequenceId',
            width: 60,
            className: 'tableHead',
            render: (text: string, record: TableObj) => {
                return (
                    <span>#{record.sequenceId}</span>
                );
            }
        }, {
            title: 'Id',
            dataIndex: 'id',
            key: 'id',
            width: 150,
            className: 'tableHead'
        }, {
            title: 'Duration',
            dataIndex: 'duration',
            key: 'duration',
            width: 150,
            render: (text: string, record: TableObj) => {
                let duration;
                if (record.duration) {
                    duration = convertDuration(record.duration);
                }
                return (
                    <div>{duration}</div>
                );
            },
        }, {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: 150,
            className: 'tableStatus',
            render: (text: string, record: TableObj) => {
                bgColor = record.status;
                return (
                    <div className={`${bgColor} commonStyle`}>
                        {record.status}
                    </div>
                );
            }
        }, {
            title: 'Default Metric',
            dataIndex: 'acc',
            key: 'acc',
            render: (text: string, record: TableObj) => {
                return (
                    <div>
                        {
                            record.acc
                                ?
                                record.acc.toFixed(6)
                                :
                                record.acc
                        }
                    </div>
                );
            }
            // width: 150
        }];

        const openRow = (record: TableObj) => {
            let isHasParameters = true;
            if (record.description.parameters.error) {
                isHasParameters = false;
            }
            const openRowDataSource = {
                parameters: record.description.parameters
            };
            let isLogLink: boolean = false;
            const logPathRow = record.description.logPath;
            if (record.description.isLink !== undefined) {
                isLogLink = true;
            }
            return (
                <pre id="description" className="hyperpar">
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
                    {
                        isLogLink
                            ?
                            <div className="logpath">
                                <span className="logName">logPath: </span>
                                <a className="logContent logHref" href={logPathRow} target="_blank">{logPathRow}</a>
                            </div>
                            :
                            <div className="logpath">
                                <span className="logName">logPath: </span>
                                <span className="logContent">{logPathRow}</span>
                            </div>
                    }
                </pre>
            );
        };
        return (
            <div className="tabScroll">
                <Table
                    columns={columns}
                    expandedRowRender={openRow}
                    dataSource={tableSource}
                    className="commonTableStyle"
                    pagination={false}
                />
            </div>
        );
    }
}

export default SuccessTable;