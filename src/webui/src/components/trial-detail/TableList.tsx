import * as React from 'react';
import axios from 'axios';
import JSONTree from 'react-json-tree';
import ReactEcharts from 'echarts-for-react';
import { Row, Table, Button, Popconfirm, Modal, message } from 'antd';
import { MANAGER_IP, trialJobStatus } from '../../static/const';
import { convertDuration } from '../../static/function';
import { TableObj, TrialJob } from '../../static/interface';
require('../../static/style/tableStatus.css');
require('../../static/style/logPath.scss');
require('../../static/style/search.scss');
require('../../static/style/table.scss');
require('../../static/style/button.scss');
const echarts = require('echarts/lib/echarts');
require('echarts/lib/chart/line');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});

interface TableListProps {
    tableSource: Array<TableObj>;
    updateList: Function;
}

interface TableListState {
    intermediateOption: object;
    modalVisible: boolean;
}

class TableList extends React.Component<TableListProps, TableListState> {

    public _isMounted = false;
    constructor(props: TableListProps) {
        super(props);

        this.state = {
            intermediateOption: {},
            modalVisible: false
        };
    }

    showIntermediateModal = (id: string) => {

        axios(`${MANAGER_IP}/metric-data/${id}`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const intermediateArr: number[] = [];
                    Object.keys(res.data).map(item => {
                        intermediateArr.push(parseFloat(res.data[item].data));
                    });
                    const intermediate = this.intermediateGraphOption(intermediateArr, id);
                    if (this._isMounted) {
                        this.setState(() => ({
                            intermediateOption: intermediate
                        }));
                    }
                }
            });
        if (this._isMounted) {
            this.setState({
                modalVisible: true
            });
        }
    }

    hideIntermediateModal = () => {
        if (this._isMounted) {
            this.setState({
                modalVisible: false
            });
        }
    }

    intermediateGraphOption = (intermediateArr: number[], id: string) => {
        const sequence: number[] = [];
        const lengthInter = intermediateArr.length;
        for (let i = 1; i <= lengthInter; i++) {
            sequence.push(i);
        }
        return {
            title: {
                text: id,
                left: 'center',
                textStyle: {
                    fontSize: 16,
                    color: '#333',
                }
            },
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                data: sequence
            },
            yAxis: {
                name: 'Default Metric',
                type: 'value',
                data: intermediateArr
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: intermediateArr
            }]
        };
    }

    // kill job
    killJob = (key: number, id: string, status: string) => {
        const { updateList } = this.props;
        axios(`${MANAGER_IP}/trial-jobs/${id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            }
        })
            .then(res => {
                if (res.status === 200) {
                    message.success('Cancel the job successfully');
                    // render the table
                    updateList();
                } else {
                    message.error('fail to cancel the job');
                }
            })
            .catch(error => {
                if (error.response.status === 500) {
                    if (error.response.data.error) {
                        message.error(error.response.data.error);
                    } else {
                        message.error('500 error, fail to cancel the job');
                    }
                }
            });
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {

        const { tableSource } = this.props;
        const { intermediateOption, modalVisible } = this.state;
        let bgColor = '';
        const trialJob: Array<TrialJob> = [];
        trialJobStatus.map(item => {
            trialJob.push({
                text: item,
                value: item
            });
        });
        const columns = [{
            title: 'Trial No.',
            dataIndex: 'sequenceId',
            key: 'sequenceId',
            width: 120,
            className: 'tableHead',
            sorter: (a: TableObj, b: TableObj) => (a.sequenceId as number) - (b.sequenceId as number)
        }, {
            title: 'Id',
            dataIndex: 'id',
            key: 'id',
            width: 60,
            className: 'tableHead idtitle',
            // the sort of string
            sorter: (a: TableObj, b: TableObj): number => a.id.localeCompare(b.id),
            render: (text: string, record: TableObj) => {
                return (
                    <div>{record.id}</div>
                );
            }
        }, {
            title: 'Duration',
            dataIndex: 'duration',
            key: 'duration',
            width: 140,
            // the sort of number
            sorter: (a: TableObj, b: TableObj) => (a.duration as number) - (b.duration as number),
            render: (text: string, record: TableObj) => {
                let duration;
                if (record.duration !== undefined && record.duration > 0) {
                    duration = convertDuration(record.duration);
                } else {
                    duration = 0;
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
                    <span className={`${bgColor} commonStyle`}>{record.status}</span>
                );
            },
            filters: trialJob,
            onFilter: (value: string, record: TableObj) => record.status.indexOf(value) === 0,
            sorter: (a: TableObj, b: TableObj): number => a.status.localeCompare(b.status)
        }, {
            title: 'Default Metric',
            dataIndex: 'acc',
            key: 'acc',
            width: 200,
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
        }, {
            title: 'Operation',
            dataIndex: 'operation',
            key: 'operation',
            width: 90,
            render: (text: string, record: TableObj) => {
                let trialStatus = record.status;
                let flagKill = false;
                if (trialStatus === 'RUNNING') {
                    flagKill = true;
                } else {
                    flagKill = false;
                }
                return (
                    flagKill
                        ?
                        (
                            <Popconfirm
                                title="Are you sure to delete this trial?"
                                onConfirm={this.killJob.bind(this, record.key, record.id, record.status)}
                            >
                                <Button type="primary" className="tableButton">Kill</Button>
                            </Popconfirm>
                        )
                        :
                        (
                            <Button
                                type="primary"
                                className="tableButton"
                                disabled={true}
                            >
                                Kill
                            </Button>
                        )
                );
            },
        }, {
            title: 'Intermediate Result',
            dataIndex: 'intermediate',
            key: 'intermediate',
            width: '16%',
            render: (text: string, record: TableObj) => {
                return (
                    <Button
                        type="primary"
                        className="tableButton"
                        onClick={this.showIntermediateModal.bind(this, record.id)}
                    >
                        Intermediate
                    </Button>
                );
            },
        }
        ];

        const openRow = (record: TableObj) => {
            let isHasParameters = true;
            if (record.description.parameters.error) {
                isHasParameters = false;
            }
            const parametersRow = {
                parameters: record.description.parameters
            };
            const intermediate = record.description.intermediate;
            let showIntermediate = '';
            if (intermediate && intermediate.length > 0) {
                showIntermediate = intermediate.join(', ');
            }
            let isLogLink: boolean = false;
            const logPathRow = record.description.logPath;
            if (record.description.isLink !== undefined) {
                isLogLink = true;
            }
            return (
                <pre id="allList" className="hyperpar">
                    {
                        isHasParameters
                            ?
                            < JSONTree
                                hideRoot={true}
                                shouldExpandNode={() => true}  // default expandNode
                                getItemString={() => (<span />)}  // remove the {} items
                                data={parametersRow}
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
                    <Row className="intermediate logpath">
                        <span className="logName">Intermediate Result:</span> 
                        {showIntermediate}
                    </Row>
                </pre>
            );
        };

        return (
            <Row className="tableList">
                <div id="tableList">
                    <Table
                        columns={columns}
                        expandedRowRender={openRow}
                        dataSource={tableSource}
                        className="commonTableStyle"
                        pagination={{ pageSize: 20 }}
                    />
                    <Modal
                        title="Intermediate Result"
                        visible={modalVisible}
                        onCancel={this.hideIntermediateModal}
                        footer={null}
                        destroyOnClose={true}
                        width="80%"
                    >
                        <ReactEcharts
                            option={intermediateOption}
                            style={{
                                width: '100%',
                                height: 0.7 * window.innerHeight
                            }}
                            theme="my_theme"
                        />
                    </Modal>
                </div>
            </Row>
        );
    }
}

export default TableList;