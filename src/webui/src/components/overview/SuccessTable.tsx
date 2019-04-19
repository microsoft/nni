import * as React from 'react';
import { Table } from 'antd';
import OpenRow from '../public-child/OpenRow';
import DefaultMetric from '../public-child/DefaultMetrc';
import { TableObj } from '../../static/interface';
import { convertDuration } from '../../static/function';
import '../../static/style/tableStatus.css';
import '../../static/style/openRow.scss';

interface SuccessTableProps {
    tableSource: Array<TableObj>;
    trainingPlatform: string;
    logCollection: boolean;
    multiphase: boolean;
}

class SuccessTable extends React.Component<SuccessTableProps, {}> {

    public _isMounted = false;

    constructor(props: SuccessTableProps) {
        super(props);

    }

    openRow = (record: TableObj) => {
        const { trainingPlatform, logCollection, multiphase } = this.props;
        return (
            <OpenRow
                trainingPlatform={trainingPlatform}
                record={record}
                logCollection={logCollection}
                multiphase={multiphase}
            />
        );
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
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
            title: 'ID',
            dataIndex: 'id',
            key: 'id',
            width: 60,
            className: 'tableHead leftTitle',
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
            title: 'Default metric',
            dataIndex: 'acc',
            key: 'acc',
            sorter: (a: TableObj, b: TableObj) => {
                if (a.acc !== undefined && b.acc !== undefined) {
                    return JSON.parse(a.acc.default) - JSON.parse(b.acc.default);
                } else {
                    return NaN;
                }
            },
            render: (text: string, record: TableObj) => {
                return (
                    <DefaultMetric record={record} />
                );
            }
        }];
        return (
            <div className="tabScroll" >
                <Table
                    columns={columns}
                    expandedRowRender={this.openRow}
                    dataSource={tableSource}
                    className="commonTableStyle"
                    pagination={false}
                />
            </div >
        );
    }
}

export default SuccessTable;
