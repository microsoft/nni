import React, { useState, useEffect } from 'react';
import { Stack, Dropdown, Toggle, IDropdownOption } from '@fluentui/react';
import ReactEcharts from 'echarts-for-react';
import { Trial } from '@model/trial';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { TooltipForAccuracy, EventMap } from '@static/interface';
import { reformatRetiariiParameter } from '@static/function';
import { gap15 } from '@components/fluent/ChildrenGap';
import { optimizeModeValue } from './optimizeMode';
import 'echarts/lib/chart/scatter';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

// this file is for overview page and detail page's Default metric graph

const EmptyGraph = {
    grid: {
        left: '8%'
    },
    xAxis: {
        name: 'Trial',
        type: 'category'
    },
    yAxis: {
        name: 'Default metric',
        type: 'value'
    }
};

interface DefaultPointProps {
    trialIds: string[];
    chartHeight: number;
    hasBestCurve: boolean;
    changeExpandRowIDs: Function;
}

const formatAccuracy = (accuracy: number | undefined): number => {
    if (accuracy === undefined || isNaN(accuracy) || !isFinite(accuracy)) {
        return 0;
    }
    return accuracy;
}

const DefaultPoint = (props: DefaultPointProps) => {
    const { hasBestCurve, trialIds, changeExpandRowIDs, chartHeight } = props;
    const [bestCurveEnabled, setBestCurveEnabled] = useState(false);
    // const [startY, setStartY] = useState(0); // dataZoomY
    // const [endY, setEndY] = useState(0); // dataZoomY
    const [graph, setGraph] = useState(EmptyGraph);
    const [onEvents, setonEvents] = useState({});
    const [userSelectOptimizeMode, setuserSelectOptimizeMode] = useState(optimizeModeValue(EXPERIMENT.optimizeMode) as string);
    const [userSelectAccuracyNumberKey, setuserSelectAccuracyNumberKey] = useState('default');
    const loadDefault = (ev: React.MouseEvent<HTMLElement>, checked?: boolean): void => {
        setBestCurveEnabled(checked ?? true); // ?? true是新加的
    };
    // const metricDataZoom = (e: EventMap): void => {
    //     if (e.batch !== undefined) {
    //         setStartY(e.batch[0].start !== null ? e.batch[0].start : 0);
    //         setEndY(e.batch[0].end !== null ? e.batch[0].end : 100);
    //     }
    // };
    useEffect(() => {
        generateGraph();
        console.info('ccc');
        // setonEvents({ dataZoom: metricDataZoom, click: pointClick });
        setonEvents({ click: pointClick });
        // const accNodata = graph === EmptyGraph ? 'No data' : '';
    }, [trialIds, bestCurveEnabled, userSelectAccuracyNumberKey, userSelectOptimizeMode]);

    const pointClick = (params: any): void => {
        // [hasBestCurve: true]: is detail page, otherwise, is overview page
        if (!hasBestCurve) {
            changeExpandRowIDs(params.data[2], 'chart');
        }
    };
    const generateGraphConfig = (_maxSequenceId: number): any => {
        return{
            grid: {
                left: '8%'
            },
            tooltip: {
                trigger: 'item',
                enterable: hasBestCurve,
                confine: true, // confirm always show tooltip box rather than hidden by background
                formatter: (data: TooltipForAccuracy): React.ReactNode => `
                    <div class="tooldetailAccuracy">
                        <div>Trial No.: ${data.data[0]}</div>
                        <div>Trial ID: ${data.data[2]}</div>
                        <div>${userSelectAccuracyNumberKey}: ${data.data[1]}</div>
                        <div>Parameters: <pre>${JSON.stringify(
                            reformatRetiariiParameter(data.data[3]),
                            null,
                            4
                        )}</pre></div>
                    </div>
                `
            },
            dataZoom: [
                {
                    // id: 'dataZoomY',
                    type: 'inside',
                    // moveOnMouseMove: false,
                    // rangeMode: ['percent', 'percent'],
                    yAxisIndex: [0],
                    filterMode: 'none',
                    start: 0, // percent
                    end: 100 // percent
                }
            ],
            xAxis: {
                name: 'Trial',
                type: 'category'
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true
            },
            series: undefined
    }};

    const generateScatterSeries = (trials: Trial[]): any => {
        let data;
        if (trials[0].accuracyNumberTypeDictKeys.length > 1) {
            // dict type final results
            data = trials.map(trial => [
                trial.sequenceId,
                trial.acc === undefined ? 0 : formatAccuracy(trial.acc[userSelectAccuracyNumberKey]),
                trial.id,
                trial.parameter
            ]);
        } else {
            data = trials.map(trial => [
                trial.sequenceId,
                formatAccuracy(trial.accuracy),
                trial.id,
                trial.parameter
            ]);
        }

        console.info(data);
        return {
            symbolSize: 6,
            type: 'scatter',
            data
        };
    }

    const generateBestCurveSeries = (trials: Trial[]): any => {
        let best = trials[0];
        const data = [
            [
                best.sequenceId,
                best.acc === undefined ? 0 : formatAccuracy(best.acc[userSelectAccuracyNumberKey]),
                best.id,
                best.parameter
            ]
        ];
        for (let i = 1; i < trials.length; i++) {
            const trial = trials[i];
            // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
            const delta = trial.acc![userSelectAccuracyNumberKey] - best.acc![userSelectAccuracyNumberKey];
            const better = userSelectOptimizeMode === 'minimize' ? delta < 0 : delta > 0;
            if (better) {
                data.push([
                    trial.sequenceId,
                    trial.acc === undefined ? 0 : formatAccuracy(trial.acc[userSelectAccuracyNumberKey]),
                    best.id,
                    trial.parameter
                ]);
                best = trial;
            } else {
                data.push([
                    trial.sequenceId,
                    best.acc === undefined ? 0 : formatAccuracy(best.acc[userSelectAccuracyNumberKey]),
                    best.id,
                    trial.parameter
                ]);
            }
        }

        return {
            type: 'line',
            lineStyle: { color: '#FF6600' },
            data
        };
    }

    const updateUserOptimizeMode = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            setuserSelectOptimizeMode(item.key.toString());
        }
    };

    // final result keys dropdown click event
    const updateTrialfinalResultKeys = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            setuserSelectAccuracyNumberKey(item.key.toString());
        }
    };

    const generateGraph = (): any => {
        const trials = TRIALS.getTrials(trialIds).filter(trial => trial.sortable);
        if (trials.length === 0) {
            return EmptyGraph;
        }
        const graph = generateGraphConfig(trials[trials.length - 1].sequenceId);
        if (bestCurveEnabled) {
            (graph as any).series = [generateBestCurveSeries(trials), generateScatterSeries(trials)];
        } else {
            (graph as any).series = [generateScatterSeries(trials)];
        }
        console.info(graph);
        setGraph(graph);
    }

    const trials = TRIALS.getTrials(trialIds).filter(trial => trial.sortable);
    let dictDropdown: string[] = [];
    if (trials.length > 0) {
        dictDropdown = trials[0].accuracyNumberTypeDictKeys;
    }
    return (
        <div>
            {hasBestCurve && (
                <Stack horizontal reversed tokens={gap15} className='default-metric'>
                    <Toggle label='Optimization curve' inlineLabel onChange={loadDefault} />
                    <Dropdown
                        selectedKey={userSelectOptimizeMode}
                        onChange={updateUserOptimizeMode}
                        options={[
                            { key: 'maximize', text: 'Maximize' },
                            { key: 'minimize', text: 'Minimize' }
                        ]}
                        styles={{ dropdown: { width: 100 } }}
                        className='para-filter-percent'
                    />
                    {dictDropdown.length > 1 && (
                        <Dropdown
                            selectedKey={userSelectAccuracyNumberKey}
                            onChange={updateTrialfinalResultKeys}
                            options={dictDropdown.map(item => ({ key: item, text: item }))}
                            styles={{ dropdown: { width: 100 } }}
                            className='para-filter-percent'
                        />
                    )}
                </Stack>
            )}
            <div className='default-metric-graph graph'>
                <ReactEcharts
                    option={graph}
                    style={{
                        width: '100%',
                        height: chartHeight,
                        margin: '0 auto'
                    }}
                    theme='nni_theme'
                    // notMerge={true} // update now
                    onEvents={onEvents}
                />
                <div className='default-metric-noData fontColor333'>{graph === EmptyGraph ? 'No data' : ''}</div>
            </div>
        </div>
    );
};

export default DefaultPoint;
