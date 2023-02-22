import React, { useState, useEffect, useContext } from 'react';
import { Stack, Dropdown, Toggle, IDropdownOption } from '@fluentui/react';
import ReactEcharts from 'echarts-for-react';
import { AppContext } from '@/App';
import { Trial } from '@model/trial';
import { TRIALS } from '@static/datamodel';
import { TooltipForAccuracy } from '@static/interface';
import { reformatRetiariiParameter } from '@static/function';
import { gap15 } from '@components/fluent/ChildrenGap';

import 'echarts/lib/chart/scatter';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

// this file is for overview page and detail page's Default metric graph
// TODO: test dict keys with updated chart

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
};

const generateGraphConfig = (hasBestCurve: boolean, finalKey: string): any => {
    return {
        grid: {
            left: '8%'
        },
        tooltip: {
            trigger: 'item',
            enterable: hasBestCurve,
            confine: true, // confirm always show tooltip box rather than hidden by background
            formatter: (data: TooltipForAccuracy): React.ReactNode => `
                <div class="tooldetailAccuracy">
                    <div class='trial-No'>No. ${data.data[0]}</div>
                    <div class='main'>
                    <div><span>Trial ID: </span>${data.data[2]}</div>
                    <div><span>${finalKey}: </span>${data.data[1]}</div>
                    <div><span>Parameters: </span><pre>${JSON.stringify(
                        reformatRetiariiParameter(data.data[3]),
                        null,
                        4
                    )}</pre></div>
                    <div>
                </div>
            `
        },
        dataZoom: [
            {
                type: 'inside',
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
    };
};

const generateGraph = (
    trialIds: string[],
    hasBestCurve: boolean,
    finalKey: string,
    bestCurveEnabled: boolean,
    optimizeMode: string
): any => {
    const trials = TRIALS.getTrials(trialIds).filter(trial => trial.sortable);
    if (trials.length === 0) {
        return EmptyGraph;
    }
    const graph = generateGraphConfig(hasBestCurve, finalKey);
    if (bestCurveEnabled) {
        (graph as any).series = [
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            generateBestCurveSeries(trials, finalKey, optimizeMode),
            // eslint-disable-next-line @typescript-eslint/no-use-before-define
            generateScatterSeries(trials, finalKey)
        ];
    } else {
        // eslint-disable-next-line @typescript-eslint/no-use-before-define
        (graph as any).series = [generateScatterSeries(trials, finalKey)];
    }
    return graph;
};

const generateScatterSeries = (trials: Trial[], finalKey: string): any => {
    let data;
    if (trials[0].accuracyNumberTypeDictKeys.length > 1) {
        // dict type final results
        data = trials.map(trial => [
            trial.sequenceId,
            trial.acc === undefined ? 0 : formatAccuracy(trial.acc[finalKey]),
            trial.id,
            trial.parameter
        ]);
    } else {
        data = trials.map(trial => [trial.sequenceId, formatAccuracy(trial.accuracy), trial.id, trial.parameter]);
    }

    return {
        symbolSize: 6,
        type: 'scatter',
        data
    };
};

const generateBestCurveSeries = (trials: Trial[], finalKey: string, optimizeMode: string): any => {
    let best = trials[0];
    const data = [
        [best.sequenceId, best.acc === undefined ? 0 : formatAccuracy(best.acc[finalKey]), best.id, best.parameter]
    ];
    for (let i = 1; i < trials.length; i++) {
        const trial = trials[i];
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const delta = trial.acc![finalKey] - best.acc![finalKey];
        const better = optimizeMode === 'Minimize' ? delta < 0 : delta > 0;
        if (better) {
            data.push([
                trial.sequenceId,
                trial.acc === undefined ? 0 : formatAccuracy(trial.acc[finalKey]),
                best.id,
                trial.parameter
            ]);
            best = trial;
        } else {
            data.push([
                trial.sequenceId,
                best.acc === undefined ? 0 : formatAccuracy(best.acc[finalKey]),
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
};

const DefaultPoint = (props: DefaultPointProps) => {
    const { hasBestCurve, trialIds, changeExpandRowIDs, chartHeight } = props;
    const { metricGraphMode, changeMetricGraphMode } = useContext(AppContext);
    const [bestCurveEnabled, setBestCurveEnabled] = useState(false);
    const [defaultMetricOption, setGraph] = useState(EmptyGraph);
    const [userSelectAccuracyNumberKey, setuserSelectAccuracyNumberKey] = useState('default');
    const loadingBestCurveLine = (ev: React.MouseEvent<HTMLElement>, checked?: boolean): void => {
        setBestCurveEnabled(checked ?? true);
    };

    useEffect(() => {
        setGraph(generateGraph(trialIds, hasBestCurve, userSelectAccuracyNumberKey, bestCurveEnabled, metricGraphMode));
    }, [trialIds.length, bestCurveEnabled, userSelectAccuracyNumberKey, metricGraphMode]);

    const pointClick = (params: any): void => {
        // [hasBestCurve: true]: is detail page, otherwise, is overview page
        if (!hasBestCurve) {
            changeExpandRowIDs(params.data[2], 'chart');
        }
    };

    const updateUserOptimizeMode = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            changeMetricGraphMode(item.key.toString() as 'Maximize' | 'Minimize');
        }
    };

    // final result keys dropdown click event
    const updateTrialfinalResultKeys = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            setuserSelectAccuracyNumberKey(item.key.toString());
        }
    };

    const trials = TRIALS.getTrials(trialIds).filter(trial => trial.sortable);
    let dictDropdown: string[] = [];
    if (trials.length > 0) {
        dictDropdown = trials[0].accuracyNumberTypeDictKeys;
    }

    const defaultMetricChart = React.useMemo(() => {
        return (
            <div className='default-metric-graph graph'>
                <ReactEcharts
                    option={defaultMetricOption}
                    style={{
                        width: '100%',
                        height: chartHeight,
                        margin: '0 auto'
                    }}
                    theme='nni_theme'
                    // notMerge={true} // update now
                    lazyUpdate={true}
                    onEvents={{ click: pointClick }}
                />
                <div className='default-metric-noData fontColor333'>
                    {defaultMetricOption === EmptyGraph ? 'No data' : ''}
                </div>
            </div>
        );
    }, [defaultMetricOption]);

    return (
        <div>
            {hasBestCurve && (
                <Stack horizontal reversed tokens={gap15} className='default-metric'>
                    <Toggle label='Optimization curve' inlineLabel onChange={loadingBestCurveLine} />
                    <Dropdown
                        selectedKey={metricGraphMode}
                        onChange={updateUserOptimizeMode}
                        options={[
                            { key: 'Maximize', text: 'Maximize' },
                            { key: 'Minimize', text: 'Minimize' }
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
            {defaultMetricChart}
        </div>
    );
};

export default DefaultPoint;
