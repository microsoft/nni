import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { Row, Col, Select, Button, message } from 'antd';
import { ParaObj, Dimobj, TableObj, SearchSpace } from '../../static/interface';
const Option = Select.Option;
require('echarts/lib/chart/parallel');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('echarts/lib/component/visualMap');
require('../../static/style/para.scss');
require('../../static/style/button.scss');

interface ParaState {
    option: object;
    paraBack: ParaObj;
    dimName: Array<string>;
    swapAxisArr: Array<string>;
    percent: number;
    paraNodata: string;
    max: number; // graph color bar limit
    min: number;
}

interface ParaProps {
    dataSource: Array<TableObj>;
    expSearchSpace: string;
}

message.config({
    top: 250,
    duration: 2,
});

class Para extends React.Component<ParaProps, ParaState> {

    public _isMounted = false;

    private chartMulineStyle = {
        width: '100%',
        height: 392,
        margin: '0 auto',
        padding: '0 15 10 15'
    };

    constructor(props: ParaProps) {
        super(props);
        this.state = {
            option: {},
            dimName: [],
            paraBack: {
                parallelAxis: [{
                    dim: 0,
                    name: ''
                }],
                data: []
            },
            swapAxisArr: [],
            percent: 0,
            paraNodata: '',
            min: 0,
            max: 1
        };
    }

    componentDidMount() {

        this._isMounted = true;
        this.reInit();
    }

    getParallelAxis =
        (
            dimName: Array<string>, searchRange: SearchSpace,
            accPara: Array<number>,
            eachTrialParams: Array<string>, paraYdata: number[][]
        ) => {
            if (this._isMounted) {
                this.setState(() => ({
                    dimName: dimName
                }));
            }
            const parallelAxis: Array<Dimobj> = [];
            // search space range and specific value [only number]
            for (let i = 0; i < dimName.length; i++) {
                const searchKey = searchRange[dimName[i]];
                switch (searchKey._type) {
                    case 'uniform':
                    case 'quniform':
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            max: searchKey._value[1],
                            min: searchKey._value[0]
                        });
                        break;

                    case 'randint':
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            max: searchKey._value[0] - 1,
                            min: 0
                        });
                        break;

                    case 'choice':
                        const data: Array<string> = [];
                        for (let j = 0; j < searchKey._value.length; j++) {
                            data.push(searchKey._value[j].toString());
                        }
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            type: 'category',
                            data: data,
                            boundaryGap: true,
                            axisLine: {
                                lineStyle: {
                                    type: 'dotted', // axis type,solid，dashed，dotted
                                    width: 1
                                }
                            },
                            axisTick: {
                                show: true,
                                interval: 0,
                                alignWithLabel: true,
                            },
                            axisLabel: {
                                show: true,
                                interval: 0,
                                // rotate: 30
                            },
                        });
                        break;
                    // support log distribute
                    case 'loguniform':
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            type: 'log',
                        });
                        break;

                    default:
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i]
                        });

                }
            }
            // get data for every lines. if dim is choice type, number -> toString()
            Object.keys(eachTrialParams).map(item => {
                let temp: Array<number> = [];
                for (let i = 0; i < dimName.length; i++) {
                    if ('type' in parallelAxis[i]) {
                        temp.push(
                            eachTrialParams[item][dimName[i]].toString()
                        );
                    } else {
                        temp.push(
                            eachTrialParams[item][dimName[i]]
                        );
                    }
                }
                paraYdata.push(temp);
            });
            // add acc
            Object.keys(paraYdata).map(item => {
                paraYdata[item].push(accPara[item]);
            });
            // according acc to sort ydata
            if (paraYdata.length !== 0) {
                const len = paraYdata[0].length - 1;
                paraYdata.sort((a, b) => b[len] - a[len]);
            }
            const paraData = {
                parallelAxis: parallelAxis,
                data: paraYdata
            };
            const { percent, swapAxisArr } = this.state;
            // need to cut down the data
            if (percent !== 0) {
                const linesNum = paraData.data.length;
                const len = Math.floor(linesNum * percent);
                paraData.data.length = len;
            }
            // need to swap the yAxis
            if (swapAxisArr.length >= 2) {
                this.swapGraph(paraData, swapAxisArr);
            }
            this.getOption(paraData);
        }

    hyperParaPic = (dataSource: Array<TableObj>, searchSpace: string) => {
        const accPara: Array<number> = [];
        // specific value array
        const eachTrialParams: Array<string> = [];
        const paraYdata: number[][] = [];
        // experiment interface search space obj
        const searchRange = JSON.parse(searchSpace);
        const dimName = Object.keys(searchRange);
        // trial-jobs interface list
        Object.keys(dataSource).map(item => {
            const temp = dataSource[item];
            if (temp.status === 'SUCCEEDED') {
                accPara.push(temp.acc.default);
                eachTrialParams.push(temp.description.parameters);
            }
        });
        if (this._isMounted) {
            this.setState({ max: Math.max(...accPara), min: Math.min(...accPara) }, () => {
                this.getParallelAxis(dimName, searchRange, accPara, eachTrialParams, paraYdata);
            });
        }
    }

    // get percent value number
    percentNum = (value: string) => {

        let vals = parseFloat(value);
        if (this._isMounted) {
            this.setState({ percent: vals }, () => {
                this.reInit();
            });
        }
    }

    // deal with response data into pic data
    getOption = (dataObj: ParaObj) => {
        const { max, min } = this.state;
        let parallelAxis = dataObj.parallelAxis;
        let paralleData = dataObj.data;
        let visualMapObj = {};
        if (max === min) {
            visualMapObj = {
                type: 'continuous',
                precision: 3,
                color: ['#CA0000', '#FFC400', '#90EE90']
            };
        } else {
            visualMapObj = {
                bottom: '20px',
                type: 'continuous',
                precision: 3,
                min: min,
                max: max,
                color: ['#CA0000', '#FFC400', '#90EE90']
            };
        }
        let optionown = {
            parallelAxis,
            tooltip: {
                trigger: 'item'
            },
            parallel: {
                parallelAxisDefault: {
                    tooltip: {
                        show: true
                    },
                    axisLabel: {
                        formatter: function (value: string) {
                            const length = value.length;
                            if (length > 16) {
                                const temp = value.split('');
                                for (let i = 16; i < temp.length; i += 17) {
                                    temp[i] += '\n';
                                }
                                return temp.join('');
                            } else {
                                return value;
                            }
                        }
                    },
                }
            },
            visualMap: visualMapObj,
            series: {
                type: 'parallel',
                smooth: true,
                lineStyle: {
                    width: 2
                },
                data: paralleData
            }
        };
        // please wait the data
        if (this._isMounted) {
            if (paralleData.length === 0) {
                this.setState({
                    paraNodata: 'No data'
                });
            } else {
                this.setState({
                    paraNodata: ''
                });
            }
        }
        // draw search space graph
        if (this._isMounted) {
            this.setState(() => ({
                option: optionown
            }));
        }
    }

    // get swap parallel axis
    getSwapArr = (value: Array<string>) => {

        if (this._isMounted) {
            this.setState(() => ({ swapAxisArr: value }));
        }
    }

    reInit = () => {
        const { dataSource, expSearchSpace } = this.props;
        this.hyperParaPic(dataSource, expSearchSpace);
    }

    sortDimY = (a: Dimobj, b: Dimobj) => {
        return a.dim - b.dim;
    }

    // deal with after swap data into pic
    swapGraph = (paraBack: ParaObj, swapAxisArr: string[]) => {
        const paralDim = paraBack.parallelAxis;
        const paraData = paraBack.data;
        let temp: number;
        let dim1: number;
        let dim2: number;
        let bool1: boolean = false;
        let bool2: boolean = false;
        let bool3: boolean = false;
        Object.keys(paralDim).map(item => {
            const paral = paralDim[item];
            switch (paral.name) {
                case swapAxisArr[0]:
                    dim1 = paral.dim;
                    bool1 = true;
                    break;

                case swapAxisArr[1]:
                    dim2 = paral.dim;
                    bool2 = true;
                    break;

                default:
            }
            if (bool1 && bool2) {
                bool3 = true;
            }
        });
        // swap dim's number
        Object.keys(paralDim).map(item => {
            if (bool3) {
                if (paralDim[item].name === swapAxisArr[0]) {
                    paralDim[item].dim = dim2;
                }
                if (paralDim[item].name === swapAxisArr[1]) {
                    paralDim[item].dim = dim1;
                }
            }
        });
        paralDim.sort(this.sortDimY);
        // swap data array
        Object.keys(paraData).map(paraItem => {

            temp = paraData[paraItem][dim1];
            paraData[paraItem][dim1] = paraData[paraItem][dim2];
            paraData[paraItem][dim2] = temp;
        });
    }

    componentWillReceiveProps(nextProps: ParaProps) {
        const dataSource = nextProps.dataSource;
        const expSearchSpace = nextProps.expSearchSpace;
        this.hyperParaPic(dataSource, expSearchSpace);

    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { option, paraNodata, dimName } = this.state;
        return (
            <Row className="parameter">
                <Row>
                    <Col span={6} />
                    <Col span={18}>
                        <Row className="meline">
                            <span>Top</span>
                            <Select
                                style={{ width: '20%', marginRight: 10 }}
                                placeholder="100%"
                                optionFilterProp="children"
                                onSelect={this.percentNum}
                            >
                                <Option value="0.2">20%</Option>
                                <Option value="0.5">50%</Option>
                                <Option value="0.8">80%</Option>
                                <Option value="1">100%</Option>
                            </Select>
                            <Select
                                style={{ width: '60%' }}
                                mode="multiple"
                                placeholder="Please select two items to swap"
                                onChange={this.getSwapArr}
                                maxTagCount={2}
                            >
                                {
                                    dimName.map((key, item) => {
                                        return (
                                            <Option key={key} value={dimName[item]}>{dimName[item]}</Option>
                                        );
                                    })
                                }
                            </Select>
                            <Button
                                type="primary"
                                className="changeBtu tableButton"
                                onClick={this.reInit}
                            >
                                Confirm
                            </Button>
                        </Row>
                    </Col>
                </Row>
                <Row className="searcHyper">
                    <ReactEcharts
                        option={option}
                        style={this.chartMulineStyle}
                        lazyUpdate={true}
                        notMerge={true} // update now
                    />
                    <div className="noneData">{paraNodata}</div>
                </Row>
            </Row>
        );
    }
}

export default Para;