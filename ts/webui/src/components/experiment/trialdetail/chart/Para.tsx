import React, { useState, useEffect, useContext } from 'react';
import * as d3 from 'd3';
import { Dropdown, IDropdownOption, Stack, DefaultButton } from '@fluentui/react';
import ParCoords from 'parcoord-es';
import { AppContext } from '@/App';
import { Trial } from '@model/trial';
import { SearchSpace } from '@model/searchspace';
import { getValue } from '@model/localStorage';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { SingleAxis, MultipleAxes } from '@static/interface';
import ChangeColumnComponent from '../ChangeColumnComponent';

import 'parcoord-es/dist/parcoords.css';
import '@style/button.scss';
import '@style/experiment/trialdetail/para.scss';

interface ParaProps {
    trials: Trial[];
    searchSpace: SearchSpace;
}

const chartMulineStyle = {
    width: '100%',
    height: 392,
    margin: '0 auto'
};

const innerChartMargins = {
    top: 32,
    right: 20,
    bottom: 20,
    left: 28
};

// TODO: class 全局变量，需要调试
let pcs: any;
const paraRef = React.createRef<HTMLDivElement>();

const Para = (props: ParaProps) => {
    const { metricGraphMode, changeMetricGraphMode } = useContext(AppContext);
    const { trials, searchSpace } = props;
    const [selectedPercent, setSelectedPercent] = useState('1');
    const [primaryMetricKey, setPrimaryMetricKey] = useState('default');
    const [noChart, setNoChart] = useState(true);
    const [customizeColumnsDialogVisible, setCustomizeColumnsDialogVisible] = useState(false);
    const [availableDimensions, setAvailableDimensions] = useState([] as string[]);
    const [chosenDimensions, setChosenDimensions] = useState(
        localStorage.getItem(`${EXPERIMENT.profile.id}_paraColumns`) !== null &&
            getValue(`${EXPERIMENT.profile.id}_paraColumns`) !== null
            ? // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
              JSON.parse(getValue(`${EXPERIMENT.profile.id}_paraColumns`)!)
            : []
    );

    // get percent value number
    const percentNum = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            setSelectedPercent(item.key.toString()); // 触发useeffect
        }
    };

    // get user mode number 'max' or 'min'
    const updateUserOptimizeMode = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            // setUserSelectOptimizeMode(item.key.toString()); // 原理同 percentNum function
            changeMetricGraphMode(item.key.toString() as 'Maximize' | 'Minimize');
        }
    };

    // select all final keys
    const updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            setPrimaryMetricKey(item.key);
        }
    };

    const finalKeysDropdown = (): any => {
        if (TRIALS.finalKeys().length === 1) {
            return null;
        } else {
            const finalKeysDropdown: any = [];
            TRIALS.finalKeys().forEach(item => {
                finalKeysDropdown.push({
                    key: item,
                    text: item
                });
            });
            return (
                <div>
                    <span className='para-filter-text para-filter-middle'>Metrics</span>
                    <Dropdown
                        selectedKey={primaryMetricKey}
                        options={finalKeysDropdown}
                        onChange={updateEntries}
                        styles={{ root: { width: 150, display: 'inline-block' } }}
                        className='para-filter-percent'
                    />
                </div>
            );
        }
    };

    const getTrialsAsObjectList = (inferredSearchSpace: MultipleAxes, inferredMetricSpace: MultipleAxes): {}[] => {
        return trials.map(s => {
            const entries = Array.from(s.parameters(inferredSearchSpace).entries());
            entries.push(...Array.from(s.metrics(inferredMetricSpace).entries()));
            const ret = {};
            for (const [k, v] of entries) {
                ret[k.fullName] = v;
            }
            return ret;
        });
    };

    const getRange = (): [number, number] => {
        // Documentation is lacking.
        // Reference: https://github.com/syntagmatic/parallel-coordinates/issues/308
        // const range = this.pcs.height() - this.pcs.margin().top - this.pcs.margin().bottom;
        const range = chartMulineStyle.height - innerChartMargins.top - innerChartMargins.bottom;
        return [range, 1];
    };

    const convertToD3Scale = (axis: SingleAxis, initRange: boolean = true): any => {
        const padLinear = ([x0, x1], k = 0.1): [number, number] => {
            const dx = ((x1 - x0) * k) / 2;
            return [x0 - dx, x1 + dx];
        };
        const padLog = ([x0, x1], k = 0.1): [number, number] => {
            const [y0, y1] = padLinear([Math.log(x0), Math.log(x1)], k);
            return [Math.exp(y0), Math.exp(y1)];
        };
        let scaleInst: any = undefined;
        if (axis.scale === 'ordinal') {
            if (axis.nested) {
                // TODO: handle nested entries
                scaleInst = d3.scalePoint().domain(Array.from(axis.domain.keys())).padding(0.2);
            } else {
                scaleInst = d3.scalePoint().domain(axis.domain).padding(0.2);
            }
        } else if (axis.scale === 'log') {
            scaleInst = d3.scaleLog().domain(padLog(axis.domain));
        } else if (axis.scale === 'linear') {
            scaleInst = d3.scaleLinear().domain(padLinear(axis.domain));
        }
        if (initRange) {
            scaleInst = scaleInst.range(getRange());
        }
        return scaleInst;
    };

    const _updateDisplayedColumns = (displayedColumns: string[]): void => {
        setChosenDimensions(displayedColumns);
    };
    /**
     * Render the parallel coordinates. Using trial data as base and leverage
     * information from search space at a best effort basis.
     * @param source Array of trial data
     * @param searchSpace Search space
     */
    const renderParallelCoordinates = (): void => {
        const percent = parseFloat(selectedPercent);
        const inferredSearchSpace = TRIALS.inferredSearchSpace(searchSpace);
        const inferredMetricSpace = TRIALS.inferredMetricSpace();
        let convertedTrials = getTrialsAsObjectList(inferredSearchSpace, inferredMetricSpace);

        const dimensions: [string, any][] = [];
        let colorDim: string | undefined = undefined,
            colorScale: any = undefined;
        // treat every axis as numeric to fit for brush
        for (const [k, v] of inferredSearchSpace.axes) {
            dimensions.push([
                k,
                {
                    type: 'number',
                    yscale: convertToD3Scale(v)
                }
            ]);
        }
        for (const [k, v] of inferredMetricSpace.axes) {
            const scale = convertToD3Scale(v);
            if (k === primaryMetricKey && scale !== undefined && scale.interpolate) {
                // set color for primary metrics
                // `colorScale` is used to produce a color range, while `scale` is to produce a pixel range
                colorScale = convertToD3Scale(v, false);
                // convertedTrials.sort((a, b) => (userSelectOptimizeMode === 'minimize' ? a[k] - b[k] : b[k] - a[k]));
                convertedTrials.sort((a, b) => (metricGraphMode === 'Minimize' ? a[k] - b[k] : b[k] - a[k]));
                // filter top trials
                if (percent != 1) {
                    const keptTrialNum = Math.max(Math.ceil(convertedTrials.length * percent), 1);
                    convertedTrials = convertedTrials.slice(0, keptTrialNum);
                    const domain = d3.extent(convertedTrials, item => item[k]);
                    scale.domain([domain[0], domain[1]]);
                    colorScale.domain([domain[0], domain[1]]);
                    if (colorScale !== undefined) {
                        colorScale.domain(domain);
                    }
                }
                // reverse the converted trials to show the top ones upfront
                convertedTrials.reverse();
                const assignColors = (scale: any): void => {
                    scale.range([0, 1]); // fake a range to perform invert
                    const [scaleMin, scaleMax] = scale.domain();
                    const pivot = scale.invert(0.5);
                    scale
                        .domain([scaleMin, pivot, scaleMax])
                        .range(['#90EE90', '#FFC400', '#CA0000'])
                        .interpolate(d3.interpolateHsl);
                };
                assignColors(colorScale);
                colorDim = k;
            }
            dimensions.push([
                k,
                {
                    type: 'number',
                    yscale: scale
                }
            ]);
        }

        if (convertedTrials.length === 0 || dimensions.length <= 1) {
            return;
        }

        const firstRun = pcs === undefined;
        if (firstRun) {
            pcs = ParCoords()(paraRef.current);
        }
        pcs.data(convertedTrials).dimensions(
            dimensions
                .filter(([d, _]) => chosenDimensions.length === 0 || chosenDimensions.includes(d))
                .reduce((obj, entry) => ({ ...obj, [entry[0]]: entry[1] }), {})
        );

        if (firstRun) {
            pcs.margin(innerChartMargins)
                .alphaOnBrushed(0.2)
                .smoothness(0.1)
                .brushMode('1D-axes')
                .reorderable()
                .interactive();
        }
        if (colorScale !== undefined) {
            pcs.color(d => (colorScale as any)(d[colorDim as any]));
        }
        pcs.render();
        // if (firstRun) {
        if (convertedTrials.length >= 0) {
            setNoChart(false);
        }

        // set new available dims
        setAvailableDimensions(dimensions.map(e => e[0]));
    };

    useEffect(() => {
        // FIXME: redundant update(comment for componentDidUpdate)
        renderParallelCoordinates();
        // return function clearPCS() {
        //     pcs = undefined;
        // };
        // }, [chosenDimensions, selectedPercent, userSelectOptimizeMode, primaryMetricKey, trials, searchSpace]); // 百分比变化触发页面更新
    }, [chosenDimensions, selectedPercent, metricGraphMode, primaryMetricKey, trials, searchSpace]); // 百分比变化触发页面更新

    return (
        <div className='parameter'>
            <Stack horizontal className='para-filter' horizontalAlign='end'>
                <DefaultButton
                    text='Add/Remove axes'
                    onClick={(): void => {
                        setCustomizeColumnsDialogVisible(true);
                    }}
                    styles={{ root: { marginRight: 10 } }}
                />
                <Dropdown
                    // selectedKey={userSelectOptimizeMode}
                    selectedKey={metricGraphMode}
                    onChange={updateUserOptimizeMode}
                    options={[
                        { key: 'Maximize', text: 'Maximize' },
                        { key: 'Minimize', text: 'Minimize' }
                    ]}
                    styles={{ dropdown: { width: 100 } }}
                    className='para-filter-percent'
                />
                <Dropdown
                    selectedKey={selectedPercent}
                    onChange={percentNum}
                    options={[
                        { key: '0.01', text: 'Top 1%' },
                        { key: '0.05', text: 'Top 5%' },
                        { key: '0.2', text: 'Top 20%' },
                        { key: '1', text: 'Top 100%' }
                    ]}
                    styles={{ dropdown: { width: 120 } }}
                    className='para-filter-percent'
                />
                {finalKeysDropdown()}
            </Stack>
            {customizeColumnsDialogVisible && availableDimensions.length > 0 && (
                <ChangeColumnComponent
                    selectedColumns={chosenDimensions}
                    allColumns={availableDimensions.map(dim => ({ key: dim, name: dim }))}
                    onSelectedChange={_updateDisplayedColumns}
                    // onSelectedChange={(selected: string[]): void => {
                    //     setChosenDimensions(selected);
                    // }}
                    onHideDialog={(): void => {
                        setCustomizeColumnsDialogVisible(false);
                    }}
                    minSelected={2}
                    whichComponent='para'
                />
            )}
            <div className='parcoords' style={chartMulineStyle} ref={paraRef} />
            {noChart && <div className='nodata'>No data</div>}
        </div>
    );
};

export default Para;
