import * as React from 'react';
import { Table } from 'antd';
import OpenRow from '../public-child/OpenRow';
import DefaultMetric from '../public-child/DefaultMetrc';
import { TRIALS } from '../../static/datamodel';
import { TableRecord } from '../../static/interface';
import { convertDuration } from '../../static/function';
import '../../static/style/tableStatus.css';
import '../../static/style/openRow.scss';

interface SuccessTableProps {
    trialIds: string[];
}

function openRow(record: TableRecord): any {
    return (
        <OpenRow trialId={record.id} />
    );
}

class SuccessTable extends React.Component<SuccessTableProps, {}> {
    constructor(props: SuccessTableProps) {
        super(props);
    }

    render(): React.ReactNode {
        const columns = [
            {
                title: 'Trial No.',
                dataIndex: 'sequenceId',
                className: 'tableHead'
            }, {
                title: 'ID',
                dataIndex: 'id',
                width: 80,
                className: 'tableHead leftTitle',
                render: (text: string, record: TableRecord): React.ReactNode => {
                    return (
                        <div>{record.id}</div>
                    );
                },
            }, {
                title: 'Duration',
                dataIndex: 'duration',
                width: 140,
                render: (text: string, record: TableRecord): React.ReactNode => {
                    return (
                        <div className="durationsty"><div>{convertDuration(record.duration)}</div></div>
                    );
                },
            }, {
                title: 'Status',
                dataIndex: 'status',
                width: 150,
                className: 'tableStatus',
                render: (text: string, record: TableRecord): React.ReactNode => {
                    return (
                        <div className={`${record.status} commonStyle`}>{record.status}</div>
                    );
                }
            }, {
                title: 'Default metric',
                dataIndex: 'accuracy',
                render: (text: string, record: TableRecord): React.ReactNode => {
                    return (
                        <DefaultMetric trialId={record.id} />
                    );
                }
            }
        ];
        return (
            <div className="tabScroll" >
                <Table
                    columns={columns}
                    expandedRowRender={openRow}
                    dataSource={TRIALS.table(this.props.trialIds)}
                    className="commonTableStyle"
                    pagination={false}
                />
            </div>
        );
    }
}

export default SuccessTable;
