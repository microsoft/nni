import * as React from 'react';
import { Stack, PrimaryButton, Toggle } from 'office-ui-fabric-react';
import { TooltipForIntermediate, TableObj, Intermedia } from '../../static/interface'; // eslint-disable-line no-unused-vars
import ReactEcharts from 'echarts-for-react';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

interface IntermediateState {
    detailSource: Array<TableObj>;
    interSource: object;
    filterSource: Array<TableObj>;
    eachIntermediateNum: number; // trial's intermediate number count
    isLoadconfirmBtn: boolean;
    isFilter?: boolean | undefined;
    length: number;
    clickCounts: number; // user filter intermediate click confirm btn's counts
}

interface IntermediateProps {
    source: Array<TableObj>;
    whichGraph: string;
}

class Intermediate extends React.Component<IntermediateProps, IntermediateState> {

    static intervalMediate = 1;
    public pointInput!: HTMLInputElement | null;
    public minValInput!: HTMLInputElement | null;
    public maxValInput!: HTMLInputElement | null;

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

    drawIntermediate = (source: Array<TableObj>): void => {
        if (source.length > 0) {
            this.setState({
                length: source.length,
                detailSource: source
            });
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
                    position: function (point: Array<number>, data: TooltipForIntermediate): number[] {
                        if (data.dataIndex < length / 2) {
                            return [point[0], 80];
                        } else {
                            return [point[0] - 300, 80];
                        }
                    },
                    formatter: function (data: TooltipForIntermediate): React.ReactNode {
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
                    // name: '# Intermediate',
                    boundaryGap: false,
                    data: xAxis
                },
                yAxis: {
                    type: 'value',
                    name: 'Metric',
                    scale: true,
                },
                series: trialIntermediate
            };
            this.setState({
                interSource: option
            });
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
                    name: 'Metric'
                }
            };
            this.setState({ interSource: nullData });
        }
    }

    // confirm btn function [filter data]
    filterLines = (): void => {
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
                this.setState({ filterSource: filterSource });
                this.drawIntermediate(filterSource);
            }
            const counts = this.state.clickCounts + 1;
            this.setState({ isLoadconfirmBtn: false, clickCounts: counts });
        });
    }

    switchTurn = (ev: React.MouseEvent<HTMLElement>, checked?: boolean): void => {
        this.setState({ isFilter: checked });
        if (checked === false) {
            this.drawIntermediate(this.props.source);
        }
    }

    componentDidMount(): void {
        const { source } = this.props;
        this.drawIntermediate(source);
    }

    componentWillReceiveProps(nextProps: IntermediateProps, nextState: IntermediateState): void {
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

    // shouldComponentUpdate(nextProps: IntermediateProps, nextState: IntermediateState) {
    //     const { whichGraph, source } = nextProps;
    //     const beforeGraph = this.props.whichGraph;
    //     if (whichGraph === '4') {
    //         const { isFilter, length, clickCounts } = nextState;
    //         const beforeLength = this.state.length;
    //         const beforeSource = this.props.source;
    //         const beforeClickCounts = this.state.clickCounts;

    //         if (isFilter !== this.state.isFilter) {
    //             return true;
    //         }

    //         if (clickCounts !== beforeClickCounts) {
    //             return true;
    //         }

    //         if (isFilter === false) {
    //             if (whichGraph !== beforeGraph) {
    //                 return true;
    //             }
    //             if (length !== beforeLength) {
    //                 return true;
    //             }
    //             if (beforeSource.length !== source.length) {
    //                 return true;
    //             }
    //             if (beforeSource[beforeSource.length - 1] !== undefined) {
    //                 if (source[source.length - 1].description.intermediate.length !==
    //                     beforeSource[beforeSource.length - 1].description.intermediate.length) {
    //                     return true;
    //                 }
    //                 if (source[source.length - 1].duration !== beforeSource[beforeSource.length - 1].duration) {
    //                     return true;
    //                 }
    //                 if (source[source.length - 1].status !== beforeSource[beforeSource.length - 1].status) {
    //                     return true;
    //                 }
    //             }
    //         }
    //     }

    //     return false;
    // }

    render(): React.ReactNode {
        const { interSource, isLoadconfirmBtn, isFilter } = this.state;
        return (
            <div>
                {/* style in para.scss */}
                <Stack horizontal horizontalAlign="end" className="meline intermediate">
                    {
                        isFilter
                            ?
                            <span style={{ marginRight: 15 }}>
                                <span className="filter-x"># Intermediate result</span>
                                <input
                                    // placeholder="point"
                                    ref={(input): any => this.pointInput = input}
                                    className="strange"
                                />
                                <span>Metric range</span>
                                <input
                                    // placeholder="range"
                                    ref={(input): any => this.minValInput = input}
                                />
                                <span className="hyphen">-</span>
                                <input
                                    // placeholder="range"
                                    ref={(input): any => this.maxValInput = input}
                                />
                                <PrimaryButton
                                    text="Confirm"
                                    onClick={this.filterLines}
                                    disabled={isLoadconfirmBtn}
                                />
                            </span>
                            :
                            null
                    }
                    {/* filter message */}
                    <span>Filter</span>
                    <Toggle onChange={this.switchTurn} />
                </Stack>
                <div className="intermediate-graph">
                    <ReactEcharts
                        option={interSource}
                        style={{ width: '100%', height: 418, margin: '0 auto' }}
                        notMerge={true} // update now
                    />
                    <div className="yAxis"># Intermediate result</div>
                </div>
            </div>
        );
    }
}

export default Intermediate;
