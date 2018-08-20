import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../const';
import ReactEcharts from 'echarts-for-react';
import { Select, Button, message } from 'antd';
const Option = Select.Option;
require('echarts/lib/chart/parallel');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('echarts/lib/component/visualMap');
require('../style/para.css');

const chartMulineStyle = {
    width: '100%',
    height: 600,
    margin: '0 auto',
    padding: 15
};

interface Dimobj {
    dim: number;
    name: string;
    max?: number;
    min?: number;
    type?: string;
    data?: string[];
}

interface HoverName {
    name: string;
}

interface ParaObj {
    data: number[][];
    parallelAxis: Array<Dimobj>;
}

interface ParaState {
    option: object;
    paraBack: ParaObj;
    dimName: Array<string>;
    swapAxisArr: Array<string>;
    percent: number;
    paraNodata: string;
}

message.config({
    top: 250,
    duration: 2,
});

class Para extends React.Component<{}, ParaState> {

    public intervalIDPara = 4;
    public _isMounted = false;

    constructor(props: {}) {
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
        };
    }

    hyperParaPic = () => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/experiment`)
            ])
            .then(axios.spread((res, res1) => {
                if (res.status === 200 && res1.status === 200) {
                    if (res.data.length !== 0) {
                        const accParaData = res.data;
                        const accPara: Array<number> = [];
                        // specific value array
                        const speValue: Array<string> = [];
                        // yAxis specific name
                        const speDimName: Array<string> = [];
                        const parallelAxis: Array<Dimobj> = [];
                        const paraYdata: number[][] = [];
                        Object.keys(accParaData).map(item => {
                            if (accParaData[item].hyperParameters !== undefined) {
                                const tem = JSON.parse(accParaData[item].hyperParameters).parameters;
                                // get dim and every line specific number
                                speDimName.push(tem);
                            }
                            if (accParaData[item].status === 'SUCCEEDED') {
                                if (accParaData[item].finalMetricData !== undefined) {
                                    // get acc array
                                    accPara.push(parseFloat(accParaData[item].finalMetricData.data));
                                    // get dim and every line specific number
                                    const temp = JSON.parse(accParaData[item].hyperParameters).parameters;
                                    speValue.push(temp);
                                }
                            }
                        });
                        // get [batch_size...] name, default each trial is same
                        // if (speValue.length !== 0) {
                        const dimName = Object.keys(speDimName[0]);
                        if (this._isMounted) {
                            this.setState(() => ({
                                dimName: dimName
                            }));
                        }
                        // search space range and specific value [only number]
                        const searchRange = JSON.parse(res1.data.params.searchSpace);
                        for (let i = 0; i < dimName.length; i++) {
                            const searchKey = searchRange[dimName[i]];
                            if (searchKey._type === 'uniform') {
                                parallelAxis.push({
                                    dim: i,
                                    name: dimName[i],
                                    max: searchKey._value[1],
                                    min: searchKey._value[0]
                                });
                            } else { // choice
                                // data number ['0.2', '0.4', '0.6']
                                const data: Array<string> = [];
                                for (let j = 0; j < searchKey._value.length; j++) {
                                    data.push(searchKey._value[j].toString());
                                }
                                parallelAxis.push({
                                    dim: i,
                                    name: dimName[i],
                                    type: 'category',
                                    data: data
                                });
                            }
                        }
                        // get data for every lines. if dim is choice type
                        // number -> toString()
                        Object.keys(speValue).map(item => {
                            let temp: Array<number> = [];
                            for (let i = 0; i < dimName.length; i++) {
                                if ('type' in parallelAxis[i]) {
                                    temp.push(
                                        speValue[item][dimName[i]].toString()
                                    );
                                } else {
                                    temp.push(
                                        speValue[item][dimName[i]]
                                    );
                                }
                            }
                            paraYdata.push(temp);
                        });
                        // add acc
                        Object.keys(paraYdata).map(item => {
                            paraYdata[item].push(accPara[item]);
                        });
                        this.setState(() => ({
                            paraBack: {
                                parallelAxis: parallelAxis,
                                data: paraYdata
                            }
                        }));
                        const { percent, swapAxisArr } = this.state;
                        const { paraBack } = this.state;
                        // need to cut down the data
                        if (percent !== 0) {
                            const linesNum = paraBack.data.length;
                            const len = Math.floor(linesNum * percent);
                            paraBack.data.length = len;
                        }
                        // need to swap the yAxis
                        if (swapAxisArr.length >= 2) {
                            this.swapGraph(paraBack, swapAxisArr);
                        }
                        // draw search space graph
                        if (this._isMounted) {
                            this.setState({
                                option: this.getOption(paraBack)
                            });
                        }
                        // }
                    }
                }
            }));
    }

    // get percent value number
    percentNum = (value: string) => {

        window.clearInterval(this.intervalIDPara);
        let vals = parseFloat(value);
        if (this._isMounted) {
            this.setState(() => ({
                percent: vals
            }));
        }
        this.hyperParaPic();
        this.intervalIDPara = window.setInterval(this.hyperParaPic, 10000);
    }

    // deal with response data into pic data
    getOption = (dataObj: ParaObj) => {
        let parallelAxis = dataObj.parallelAxis;
        let paralleData = dataObj.data;
        let optionown = {
            parallelAxis,
            tooltip: {
                trigger: 'item',
                formatter: function (params: HoverName) {
                    return params.name;
                }
            },
            toolbox: {
                show: true,
                left: 'right',
                iconStyle: {
                    normal: {
                        borderColor: '#ddd'
                    }
                },
                feature: {
                },
                z: 202
            },
            parallel: {
                parallelAxisDefault: {
                    tooltip: {
                        show: true
                    }
                }
            },
            visualMap: {
                type: 'continuous',
                min: 0,
                max: 1,
                realtime: false,
                calculable: true,
                precision: 1,
                // gradient color
                color: ['#fb7c7c', 'yellow', 'lightblue']
            },
            highlight: {
                type: 'highlight'
            },
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
        return optionown;
    }

    // get swap parallel axis
    getSwapArr = (value: Array<string>) => {

        if (this._isMounted) {
            this.setState(() => ({ swapAxisArr: value }));
        }
    }

    swapBtn = () => {

        window.clearInterval(this.intervalIDPara);
        this.hyperParaPic();
        this.intervalIDPara = window.setInterval(this.hyperParaPic, 10000);
    }

    sortDimY = (a: Dimobj, b: Dimobj) => {

        return a.dim - b.dim;
    }

    // deal with after swap data into pic
    swapGraph = (paraBack: ParaObj, swapAxisArr: string[]) => {

        if (swapAxisArr.length >= 2) {
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
                    if (paralDim[item].name === this.state.swapAxisArr[0]) {
                        paralDim[item].dim = dim2;
                    }
                    if (paralDim[item].name === this.state.swapAxisArr[1]) {
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
    }

    componentDidMount() {

        this._isMounted = true;
        // default draw all data pic
        this.hyperParaPic();
        this.intervalIDPara = window.setInterval(this.hyperParaPic, 10000);
    }

    componentWillUnmount() {

        this._isMounted = false;
        window.clearInterval(this.intervalIDPara);
    }
    render() {
        const { option, paraNodata, dimName } = this.state;
        return (
            <div className="para">
                <div className="paraCon">
                    <div className="paraTitle">
                        <div className="paraLeft">Hyper Parameter</div>
                        <div className="paraRight">
                            <Select
                                className="parapercent"
                                style={{ width: '20%' }}
                                placeholder="100%"
                                optionFilterProp="children"
                                onSelect={this.percentNum}
                            >
                                <Option value="0.2">0.2</Option>
                                <Option value="0.5">0.5</Option>
                                <Option value="0.8">0.8</Option>
                                <Option value="1">1</Option>
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
                                className="changeBtu"
                                onClick={this.swapBtn}
                            >
                                sure
                            </Button>
                        </div>
                    </div>
                    <div className="paraGra">
                        <ReactEcharts
                            className="testt"
                            option={option}
                            style={chartMulineStyle}
                            // lazyUpdate={true}
                            notMerge={true} // update now
                        />
                        <div className="paraNodata">{paraNodata}</div>
                    </div>
                </div>
            </div>
        );
    }
}

export default Para;