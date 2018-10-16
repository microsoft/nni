import * as React from 'react';
import { browserHistory } from 'react-router';
import axios from 'axios';
import { Table, Button, Popconfirm, message, Modal } from 'antd';
import { MANAGER_IP, trialJobStatus } from '../const';
import JSONTree from 'react-json-tree';
import ReactEcharts from 'echarts-for-react';
const echarts = require('echarts/lib/echarts');
require('echarts/lib/chart/bar');
require('echarts/lib/chart/line');
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('../style/trialStatus.css');
require('../style/logPath.css');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});

interface ErrorPara {
    error?: string;
}

interface DescObj {
    parameters: ErrorPara;
    logPath?: string;
    isLink?: boolean;
}

interface TableObj {
    key: number;
    id: string;
    duration?: number;
    start: string;
    status: string;
    description: DescObj;
    end?: string;
    acc?: number;
}

interface Runtrial {
    trialId: Array<string>;
    trialTime: Array<number>;
}

interface TrialJob {
    text: string;
    value: string;
}

interface TabState {
    tableData: Array<TableObj>;
    downhref: string;
    option: object;
    trialJobs: object;
    intermediateOption: object;
    modalVisible: boolean;
}

class TrialStatus extends React.Component<{}, TabState> {

    public intervalID = 0;
    public intervalIDS = 1;
    public _isMounted = false;

    constructor(props: {}) {

        super(props);
        this.state = {
            tableData: [{
                key: 0,
                id: '',
                duration: 0,
                start: '',
                end: '',
                status: '',
                acc: 0,
                description: {
                    parameters: {}
                }
            }],
            downhref: 'javascript:;',
            option: {},
            intermediateOption: {},
            trialJobs: {},
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
                    if (this._isMounted) {
                        this.setState({
                            intermediateOption: this.intermediateGraphOption(intermediateArr, id)
                        });
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

    getOption = (dataObj: Runtrial) => {
        let xAxis = dataObj.trialTime;
        let yAxis = dataObj.trialId;
        let option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            title: {
                left: 'center',
                text: 'Trial Duration',
                textStyle: {
                    fontSize: 18,
                    color: '#333'
                }
            },
            grid: {
                bottom: '3%',
                containLabel: true,
                left: '1%',
                right: '4%'
            },
            dataZoom: [{
                type: 'slider',
                name: 'trial',
                filterMode: 'filter',
                yAxisIndex: 0,
                orient: 'vertical'
            }, {
                type: 'slider',
                name: 'trial',
                filterMode: 'filter',
                xAxisIndex: 0
            }],
            xAxis: {
                name: 'Time',
                type: 'value',
            },
            yAxis: {
                name: 'Trial',
                type: 'category',
                data: yAxis
            },
            series: [{
                type: 'bar',
                data: xAxis
            }]
        };
        return option;
    }

    drawRunGraph = () => {

        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const trialId: Array<string> = [];
                    const trialTime: Array<number> = [];
                    const trialRun: Array<Runtrial> = [];
                    Object.keys(trialJobs).map(item => {
                        if (trialJobs[item].status !== 'WAITING') {
                            let duration: number = 0;
                            const end = trialJobs[item].endTime;
                            const start = trialJobs[item].startTime;
                            if (start && end) {
                                duration = (end - start) / 1000;
                            } else {
                                duration = (new Date().getTime() - start) / 1000;
                            }
                            trialId.push(trialJobs[item].id);
                            trialTime.push(duration);
                        }
                    });
                    trialRun.push({
                        trialId: trialId,
                        trialTime: trialTime
                    });
                    if (this._isMounted && res.status === 200) {
                        this.setState({
                            option: this.getOption(trialRun[0])
                        });
                    }
                }
            });
    }

    drawTable = () => {

        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const trialTable: Array<TableObj> = [];
                    Object.keys(trialJobs).map(item => {
                        // only succeeded trials have finalMetricData
                        let desc: DescObj = {
                            parameters: {}
                        };
                        let acc = 0;
                        let duration = 0;
                        const id = trialJobs[item].id !== undefined
                            ? trialJobs[item].id
                            : '';
                        const status = trialJobs[item].status !== undefined
                            ? trialJobs[item].status
                            : '';
                        const startTime = trialJobs[item].startTime !== undefined
                            ? new Date(trialJobs[item].startTime).toLocaleString('en-US')
                            : '';
                        const endTime = trialJobs[item].endTime !== undefined
                            ? new Date(trialJobs[item].endTime).toLocaleString('en-US')
                            : '';
                        if (trialJobs[item].hyperParameters !== undefined) {
                            desc.parameters = JSON.parse(trialJobs[item].hyperParameters).parameters;
                        } else {
                            desc.parameters = { error: 'This trial\'s parameters are not available.' };
                        }
                        if (trialJobs[item].logPath !== undefined) {
                            desc.logPath = trialJobs[item].logPath;
                            const isHyperLink = /^http/gi.test(trialJobs[item].logPath);
                            if (isHyperLink) {
                                desc.isLink = true;
                            }
                        }
                        if (trialJobs[item].finalMetricData !== undefined) {
                            acc = parseFloat(trialJobs[item].finalMetricData.data);
                        }
                        if (startTime !== '' && endTime !== '') {
                            duration = (trialJobs[item].endTime - trialJobs[item].startTime) / 1000;
                        } else if (startTime !== '' && endTime === '') {
                            duration = (new Date().getTime() - trialJobs[item].startTime) / 1000;
                        } else {
                            duration = 0;
                        }
                        trialTable.push({
                            key: trialTable.length,
                            id: id,
                            status: status,
                            start: startTime,
                            end: endTime,
                            duration: duration,
                            acc: acc,
                            description: desc
                        });
                    });
                    if (this._isMounted) {
                        this.setState(() => ({
                            tableData: trialTable
                        }));
                    }
                }
            });
    }

    // kill job
    killJob = (key: number, id: string, status: string) => {

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
                    this.drawTable();
                } else {
                    message.error('fail to cancel the job');
                }
            })
            .catch(error => {
                if (error.response.status === 500) {
                    message.error('500 error, fail to cancel the job');
                }
            });
    }

    // get tensorflow address
    getTensorpage = (id: string) => {

        let path = {
            pathname: '/tensor',
            state: id
        };

        browserHistory.push(path);
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
                name: 'Accuracy',
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

    componentDidMount() {

        this._isMounted = true;
        // the init of running chart
        this.drawRunGraph();
        // the init of trials status in the table
        this.drawTable();
        this.intervalID = window.setInterval(this.drawRunGraph, 10000);
        this.intervalIDS = window.setInterval(this.drawTable, 10000);
    }

    componentWillUnmount() {

        this._isMounted = false;
        window.clearInterval(this.intervalID);
        window.clearInterval(this.intervalIDS);
    }

    render() {
        const { intermediateOption, modalVisible, option, tableData } = this.state;
        let bgColor = '';
        const trialJob: Array<TrialJob> = [];
        trialJobStatus.map(item => {
            trialJob.push({
                text: item,
                value: item
            });
        });
        const columns = [{
            title: 'Id',
            dataIndex: 'id',
            key: 'id',
            width: '10%',
            className: 'tableHead',
            // the sort of string
            sorter: (a: TableObj, b: TableObj): number => a.id.localeCompare(b.id)
        }, {
            title: 'Duration/s',
            dataIndex: 'duration',
            key: 'duration',
            width: '10%',
            // the sort of number
            sorter: (a: TableObj, b: TableObj) => (a.duration as number) - (b.duration as number)
        }, {
            title: 'Start',
            dataIndex: 'start',
            key: 'start',
            width: '15%',
            sorter: (a: TableObj, b: TableObj): number => (Date.parse(a.start) - Date.parse(b.start))
        }, {
            title: 'End',
            dataIndex: 'end',
            key: 'end',
            width: '15%',
            sorter: (a: TableObj, b: TableObj): number => (a.end as string).localeCompare(b.end as string)
        }, {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: '10%',
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
            title: 'Loss/Accuracy',
            dataIndex: 'acc',
            key: 'acc',
            width: '10%',
            sorter: (a: TableObj, b: TableObj) => (a.acc as number) - (b.acc as number)
        }, {
            title: 'Operation',
            dataIndex: 'operation',
            key: 'operation',
            width: '10%',
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
            title: 'Tensor',
            dataIndex: 'tensor',
            key: 'tensor',
            width: '16%',
            render: (text: string, record: TableObj) => {
                return (
                    <Button
                        type="primary"
                        className="tableButton"
                        onClick={this.getTensorpage.bind(this, record.id)}
                    >
                        TensorBoard
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
            let isLogLink: boolean = false;
            const logPathRow = record.description.logPath;
            if (record.description.isLink !== undefined) {
                isLogLink = true;
            }
            return (
                <pre className="hyperpar">
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
                    <Button
                        type="primary"
                        className="tableButton"
                        onClick={this.showIntermediateModal.bind(this, record.id)}
                    >
                        Intermediate Result
                    </Button>
                </pre>
            );
        };

        return (
            <div className="hyper" id="tableCenter">
                <ReactEcharts
                    option={option}
                    style={{ width: '100%', height: 600, marginBottom: 15 }}
                    theme="my_theme"
                />
                <Table
                    columns={columns}
                    expandedRowRender={openRow}
                    dataSource={tableData}
                    pagination={{ pageSize: 10 }}
                    className="tables"
                    bordered={true}
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
        );
    }
}

export default TrialStatus;