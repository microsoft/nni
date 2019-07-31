import * as React from 'react';
import { Switch } from 'antd';
import ReactEcharts from 'echarts-for-react';
import { filterByStatus } from '../../static/function';
import { TableObj, DetailAccurPoint, TooltipForAccuracy } from '../../static/interface';
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface DefaultPointProps {
    showSource: Array<TableObj>;
    height: number;
    whichGraph: string;
    optimize: string;
}

interface DefaultPointState {
    defaultSource: object;
    accNodata: string;
    succeedTrials: number;
    isViewBestCurve: boolean;
}

class DefaultPoint extends React.Component<DefaultPointProps, DefaultPointState> {
    public _isDefaultMounted = false;

    constructor(props: DefaultPointProps) {
        super(props);
        this.state = {
            defaultSource: {},
            accNodata: '',
            succeedTrials: 10000000,
            isViewBestCurve: false
        };
    }

    defaultMetric = (succeedSource: Array<TableObj>, isCurve: boolean) => {
        const { optimize } = this.props;
        const accSource: Array<DetailAccurPoint> = [];
        const showSource: Array<TableObj> = succeedSource.filter(filterByStatus);
        const lengthOfSource = showSource.length;
        const tooltipDefault = lengthOfSource === 0 ? 'No data' : '';
        if (this._isDefaultMounted === true) {
            this.setState(() => ({
                succeedTrials: lengthOfSource,
                accNodata: tooltipDefault
            }));
        }
        if (lengthOfSource === 0) {
            const nullGraph = {
                grid: {
                    left: '8%'
                },
                xAxis: {
                    name: 'Trial',
                    type: 'category',
                },
                yAxis: {
                    name: 'Default metric',
                    type: 'value',
                }
            };
            if (this._isDefaultMounted === true) {
                this.setState(() => ({
                    defaultSource: nullGraph
                }));
            }
        } else {
            const resultList: Array<number | object>[] = [];
            const lineListDefault: Array<number> = [];
            Object.keys(showSource).map(item => {
                const temp = showSource[item];
                if (temp.acc !== undefined) {
                    if (temp.acc.default !== undefined) {
                        const searchSpace = temp.description.parameters;
                        lineListDefault.push(temp.acc.default);
                        accSource.push({
                            acc: temp.acc.default,
                            index: temp.sequenceId,
                            searchSpace: searchSpace
                        });
                    }
                }
            });
            // deal with best metric line
            const bestCurve: Array<number | object>[] = []; // best curve data source
            bestCurve.push([0, lineListDefault[0], accSource[0].searchSpace]); // push the first value
            if (optimize === 'maximize') {
                for (let i = 1; i < lineListDefault.length; i++) {
                    const val = lineListDefault[i];
                    const latest = bestCurve[bestCurve.length - 1][1];
                    if (val >= latest) {
                        bestCurve.push([i, val, accSource[i].searchSpace]);
                    } else {
                        bestCurve.push([i, latest, accSource[i].searchSpace]);
                    }
                }
            } else {
                for (let i = 1; i < lineListDefault.length; i++) {
                    const val = lineListDefault[i];
                    const latest = bestCurve[bestCurve.length - 1][1];
                    if (val <= latest) {
                        bestCurve.push([i, val, accSource[i].searchSpace]);
                    } else {
                        bestCurve.push([i, latest, accSource[i].searchSpace]);
                    }
                }
            }
            Object.keys(accSource).map(item => {
                const items = accSource[item];
                let temp: Array<number | object>;
                temp = [items.index, items.acc, items.searchSpace];
                resultList.push(temp);
            });
            // isViewBestCurve: false show default metric graph
            // isViewBestCurve: true  show best curve
            if (isCurve === true) {
                if (this._isDefaultMounted === true) {
                    this.setState(() => ({
                        defaultSource: this.drawBestcurve(bestCurve, resultList)
                    }));
                }
            } else {
                if (this._isDefaultMounted === true) {
                    this.setState(() => ({
                        defaultSource: this.drawDefaultMetric(resultList)
                    }));
                }
            }
        }
    }

    drawBestcurve = (realDefault: Array<number | object>[], resultList: Array<number | object>[]) => {
        return {
            grid: {
                left: '8%'
            },
            tooltip: {
                trigger: 'item',
                enterable: true,
                position: function (point: Array<number>, data: TooltipForAccuracy) {
                    if (data.data[0] < realDefault.length / 2) {
                        return [point[0], 80];
                    } else {
                        return [point[0] - 300, 80];
                    }
                },
                formatter: function (data: TooltipForAccuracy) {
                    const result = '<div class="tooldetailAccuracy">' +
                        '<div>Trial No.: ' + data.data[0] + '</div>' +
                        '<div>Optimization curve: ' + data.data[1] + '</div>' +
                        '<div>Parameters: ' +
                        '<pre>' + JSON.stringify(data.data[2], null, 4) + '</pre>' +
                        '</div>' +
                        '</div>';
                    return result;
                }
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: resultList
            }, {
                type: 'line',
                // smooth: true,
                lineStyle: {
                    color: '#0071BC'
                },
                data: realDefault
            }]
        };
    }

    drawDefaultMetric = (resultList: Array<number | object>[]) => {
        return {
            grid: {
                left: '8%'
            },
            tooltip: {
                trigger: 'item',
                enterable: true,
                position: function (point: Array<number>, data: TooltipForAccuracy) {
                    if (data.data[0] < resultList.length / 2) {
                        return [point[0], 80];
                    } else {
                        return [point[0] - 300, 80];
                    }
                },
                formatter: function (data: TooltipForAccuracy) {
                    const result = '<div class="tooldetailAccuracy">' +
                        '<div>Trial No.: ' + data.data[0] + '</div>' +
                        '<div>Default metric: ' + data.data[1] + '</div>' +
                        '<div>Parameters: ' +
                        '<pre>' + JSON.stringify(data.data[2], null, 4) + '</pre>' +
                        '</div>' +
                        '</div>';
                    return result;
                }
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: resultList
            }]
        };
    }

    loadDefault = (checked: boolean) => {
        // checked: true show best metric curve
        const { showSource } = this.props;
        if (this._isDefaultMounted === true) {
            this.defaultMetric(showSource, checked);
            // ** deal with data and then update view layer
            this.setState(() => ({ isViewBestCurve: checked }));
        }
    }

    // update parent component state
    componentWillReceiveProps(nextProps: DefaultPointProps) {

        const { whichGraph, showSource } = nextProps;
        const { isViewBestCurve } = this.state;
        if (whichGraph === '1') {
            this.defaultMetric(showSource, isViewBestCurve);
        }
    }

    shouldComponentUpdate(nextProps: DefaultPointProps, nextState: DefaultPointState) {
        const { whichGraph } = nextProps;
        if (whichGraph === '1') {
            const { succeedTrials, isViewBestCurve } = nextState;
            const succTrial = this.state.succeedTrials;
            const isViewBestCurveBefore = this.state.isViewBestCurve;
            if (isViewBestCurveBefore !== isViewBestCurve) {
                return true;
            }
            if (succeedTrials !== succTrial) {
                return true;
            }
        }
        // only whichGraph !== '1', default metric can't update
        return false;
    }

    componentDidMount() {
        this._isDefaultMounted = true;
    }

    componentWillUnmount() {
        this._isDefaultMounted = false;
    }

    render() {
        const { height } = this.props;
        const { defaultSource, accNodata, isViewBestCurve } = this.state;
        return (
            <div>
                <div className="default-metric">
                    <div className="position">
                        {
                            isViewBestCurve
                                ?
                                <span className="bold">Click here to show <span>default curve</span></span>
                                :
                                <span className="bold">Click here to show <span>optimization curve</span></span>
                        }
                        <Switch defaultChecked={false} onChange={this.loadDefault} />
                    </div>
                </div>
                <ReactEcharts
                    option={defaultSource}
                    style={{
                        width: '100%',
                        height: height,
                        margin: '0 auto',
                    }}
                    theme="my_theme"
                    notMerge={true} // update now
                />
                <div className="showMess">{accNodata}</div>
            </div>
        );
    }
}

export default DefaultPoint;