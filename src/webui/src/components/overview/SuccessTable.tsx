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
            title: 'Trial No.',
            dataIndex: 'sequenceId',
            key: 'sequenceId',
            width: 140,
            className: 'tableHead'
        }, {
            title: 'Id',
            dataIndex: 'id',
            key: 'id',
            width: 60,
            className: 'tableHead idtitle',
            render: (text: string, record: TableObj) => {
                return (
                    <div>{record.id}</div>
                );
            },
        }, {
            title: 'Duration',
            dataIndex: 'duration',
            key: 'duration',
            width: 140,
            sorter: (a: TableObj, b: TableObj) => (a.duration as number) - (b.duration as number),
            render: (text: string, record: TableObj) => {
                let duration;
                if (record.duration) {
                    duration = convertDuration(record.duration);
                }
                return (
                    <div className="durationsty"><div>{duration}</div></div>
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
            sorter: (a: TableObj, b: TableObj) => (a.acc as number) - (b.acc as number),
            render: (text: string, record: TableObj) => {
                const accuracy = record.acc;
                let wei = 0;
                if (accuracy) {
                    if (accuracy.toString().indexOf('.') !== -1) {
                        wei = accuracy.toString().length - accuracy.toString().indexOf('.') - 1;
                    }
                }
                return (
                    <div>
                        {
                            record.acc
                                ?
                                wei > 6
                                    ?
                                    record.acc.toFixed(6)
                                    :
                                    record.acc
                                :
                                'NaN'
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