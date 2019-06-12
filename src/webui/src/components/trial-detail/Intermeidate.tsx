import * as React from 'react';
import { Row, Col, Button, Switch } from 'antd';
import { TooltipForIntermediate, TableObj } from '../../static/interface';
import ReactEcharts from 'echarts-for-react';
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface Intermedia {
    name: string; // id
    type: string;
    data: Array<number | object>; // intermediate data
    hyperPara: object; // each trial hyperpara value
}

interface IntermediateState {
    detailSource: Array<TableObj>;
    interSource: object;
    filterSource: Array<TableObj>;
    eachIntermediateNum: number; // trial's intermediate number count
    isLoadconfirmBtn: boolean;
    isFilter: boolean;
    length: number; 
    clickCounts: number; // user filter intermediate click confirm btn's counts
}

interface IntermediateProps {
    source: Array<TableObj>;
    whichGraph: string;
}

class Intermediate extends React.Component<IntermediateProps, IntermediateState> {

    static intervalMediate = 1;
    public _isMounted = false;
    public pointInput: HTMLInputElement | null;
    public minValInput: HTMLInputElement | null;
    public maxValInput: HTMLInputElement | null;

    constructor(props: IntermediateProps) {
        super(props);
        this.state = {
            detailSource: [],
            interSource: {},
            filterSource: [],
            eachIntermediateNum: 1,
            isLoadconfirmBtn: false,
            isFilter: false,
            length: 100000,
            clickCounts: 0
        };
    }

    drawIntermediate = (source: Array<TableObj>) => {
        if (source.length > 0) {
            if (this._isMounted) {
                this.setState(() => ({
                    length: source.length,
                    detailSource: source
                }));
            }
            const trialIntermediate: Array<Intermedia> = [];
            Object.keys(source).map(item => {
                const temp = source[item];
                trialIntermediate.push({
                    name: temp.id,
                    data: temp.description.intermediate,
                    type: 'line',
                    hyperPara: temp.description.parameters
                });
            });
            // find max intermediate number
            trialIntermediate.sort((a, b) => { return (b.data.length - a.data.length); });
            const legend: Array<string> = [];
            // max length
            const length = trialIntermediate[0].data.length;
            const xAxis: Array<number> = [];
            Object.keys(trialIntermediate).map(item => {
                const temp = trialIntermediate[item];
                legend.push(temp.name);
            });
            for (let i = 1; i <= length; i++) {
                xAxis.push(i);
            }
            const option = {
                tooltip: {
                    trigger: 'item',
                    enterable: true,
                    position: function (point: Array<number>, data: TooltipForIntermediate) {
                        if (data.dataIndex < length / 2) {
                            return [point[0], 80];
                        } else {
                            return [point[0] - 300, 80];
                        }
                    },
                    formatter: function (data: TooltipForIntermediate) {
                        const trialId = data.seriesName;
                        let obj = {};
                        const temp = trialIntermediate.find(key => key.name === trialId);
                        if (temp !== undefined) {
                            obj = temp.hyperPara;
                        }
                        return '<div class="tooldetailAccuracy">' +
                            '<div>Trial ID: ' + trialId + '</div>' +
                            '<div>Intermediate: ' + data.data + '</div>' +
                            '<div>Parameters: ' +
                            '<pre>' + JSON.stringify(obj, null, 4) + '</pre>' +
                            '</div>' +
                            '</div>';
                    }
                },
                grid: {
                    left: '5%',
                    top: 40,
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    name: 'Step',
                    boundaryGap: false,
                    data: xAxis
                },
                yAxis: {
                    type: 'value',
                    name: 'metric'
                },
                series: trialIntermediate
            };
            if (this._isMounted) {
                this.setState(() => ({
                    interSource: option
                }));
            }
        } else {
            const nullData = {
                grid: {
                    left: '5%',
                    top: 40,
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                },
                yAxis: {
                    type: 'value',
                    name: 'Scape'
                }
            };
            if (this._isMounted) {
                this.setState(() => ({ interSource: nullData }));
            }
        }
    }

    // confirm btn function [filter data]
    filterLines = () => {
        if (this._isMounted) {
            const filterSource: Array<TableObj> = [];
            this.setState({ isLoadconfirmBtn: true }, () => {
                const { source } = this.props;
                // get input value
                const pointVal = this.pointInput !== null ? this.pointInput.value : '';
                const minVal = this.minValInput !== null ? this.minValInput.value : '';
                const maxVal = this.maxValInput !== null ? this.maxValInput.value : '';
                // user not input message
                if (pointVal === '' || minVal === '') {
                    alert('Please input filter message');
                } else {
                    // user not input max value
                    const position = JSON.parse(pointVal);
                    const min = JSON.parse(minVal);
                    if (maxVal === '') {
                        Object.keys(source).map(item => {
                            const temp = source[item];
                            const val = temp.description.intermediate[position - 1];
                            if (val >= min) {
                                filterSource.push(temp);
                            }
                        });
                    } else {
                        const max = JSON.parse(maxVal);
                        Object.keys(source).map(item => {
                            const temp = source[item];
                            const val = temp.description.intermediate[position - 1];
                            if (val >= min && val <= max) {
                                filterSource.push(temp);
                            }
                        });
                    }
                    if (this._isMounted) {
                        this.setState({ filterSource: filterSource });
                    }
                    this.drawIntermediate(filterSource);
                    const counts = this.state.clickCounts + 1;
                    this.setState({ isLoadconfirmBtn: false, clickCounts: counts });
                }
            });
        }
    }

    switchTurn = (checked: boolean) => {
        if (this._isMounted) {
            this.setState({ isFilter: checked });
        }
        if (checked === false) {
            this.drawIntermediate(this.props.source);
        }
    }

    componentDidMount() {
        this._isMounted = true;
        const { source } = this.props;
        this.drawIntermediate(source);
    }

    componentWillReceiveProps(nextProps: IntermediateProps, nextState: IntermediateState) {
        const { isFilter, filterSource } = nextState;
        const { whichGraph, source } = nextProps;
        
        if (whichGraph === '4') {
            if (isFilter === true) {
                const pointVal = this.pointInput !== null ? this.pointInput.value : '';
                const minVal = this.minValInput !== null ? this.minValInput.value : '';
                if (pointVal === '' && minVal === '') {
                    this.drawIntermediate(source);
                } else {
                    this.drawIntermediate(filterSource);
                }
            } else {
                this.drawIntermediate(source);
            }
        }
    }

    shouldComponentUpdate(nextProps: IntermediateProps, nextState: IntermediateState) {
        const { whichGraph } = nextProps;
        const beforeGraph = this.props.whichGraph;
        if (whichGraph === '4') {
            
            const { source } = nextProps;
            const { isFilter, length, clickCounts } = nextState;
            const beforeLength = this.state.length;
            const beforeSource = this.state.detailSource;
            const beforeClickCounts = this.state.clickCounts;
    
            if (isFilter !== this.state.isFilter) {
                return true;
            }

            if (clickCounts !== beforeClickCounts) {
                return true;
            }
            
            if (isFilter === false) {
                if (whichGraph !== beforeGraph) {
                    return true;
                }
                if (length !== beforeLength) {
                    return true;
                }
                if (source[source.length - 1].description.intermediate.length !==
                    beforeSource[beforeSource.length - 1].description.intermediate.length) {
                    return true;
                }
                if (source[source.length - 1].duration !== beforeSource[beforeSource.length - 1].duration) {
                    return true;
                }
                if (source[source.length - 1].status !== beforeSource[beforeSource.length - 1].status) {
                    return true;
                }
            }
        }

        return false;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { interSource, isLoadconfirmBtn, isFilter } = this.state;
        return (
            <div>
                {/* style in para.scss */}
                <Row className="meline intermediate">
                    <Col span={8} />
                    <Col span={3} style={{ height: 34 }}>
                        {/* filter message */}
                        <span>filter</span>
                        <Switch
                            defaultChecked={false}
                            onChange={this.switchTurn}
                        />
                    </Col>
                    {
                        isFilter
                            ?
                            <div>
                                <Col span={3}>
                                    <span>Scape</span>
                                    <input
                                        placeholder="point"
                                        ref={input => this.pointInput = input}
                                        className="strange"
                                    />
                                </Col>
                                <Col className="range" span={10}>
                                    <span>Intermediate result</span>
                                    <input
                                        placeholder="number"
                                        ref={input => this.minValInput = input}
                                    />
                                    <span className="heng">-</span>
                                    <input
                                        placeholder="number"
                                        ref={input => this.maxValInput = input}
                                    />
                                    <Button
                                        type="primary"
                                        className="changeBtu tableButton"
                                        onClick={this.filterLines}
                                        disabled={isLoadconfirmBtn}
                                    >
                                        Confirm
                                    </Button>
                                </Col>
                            </div>
                            :
                            <Col />
                    }
                </Row>
                <Row>
                    <ReactEcharts
                        option={interSource}
                        style={{ width: '100%', height: 418, margin: '0 auto' }}
                        notMerge={true} // update now
                    />
                </Row>
            </div>
        );
    }
}

export default Intermediate;