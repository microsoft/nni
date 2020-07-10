import { Dropdown, IDropdownOption, PrimaryButton, Stack } from 'office-ui-fabric-react';
import ParCoords from 'parcoord-es';
import 'parcoord-es/dist/parcoords.css';
import * as React from 'react';
import { TRIALS } from '../../static/datamodel';
import { filterByStatus } from '../../static/function';
import { ParaObj, TableObj } from '../../static/interface';
import '../../static/style/button.scss';
import '../../static/style/para.scss';

function isSearchSpaceNested(searchSpace): boolean {
    for (const item of Object.values(searchSpace)) {
        const value = (item as any)._value;
        if (value && typeof value[0] === 'object') {
            return true;
        }
    }
    return false;
}

interface ParaState {
    // paraSource: Array<TableObj>;
    option: object;
    paraBack: ParaObj;
    dimName: string[];
    swapAxisArr: string[];
    percent: number;
    paraNodata: string;
    max: number; // graph color bar limit
    min: number;
    sutrialCount: number; // succeed trial numbers for SUC
    succeedRenderCount: number; // all succeed trials number
    clickCounts: number;
    isLoadConfirm: boolean;
    // office-fabric-ui
    selectedItem?: { key: string | number | undefined }; // percent Selector
    swapyAxis?: string[]; // yAxis Selector
    paraYdataNested: number[][];
    isNested: boolean;
    showFinalMetricKey: string;
    metricType: string;
}

interface ParaProps {
    dataSource: Array<TableObj>;
    expSearchSpace: string;
    whichGraph: string;
}

class Para extends React.Component<ParaProps, ParaState> {

    private paraRef = React.createRef<HTMLDivElement>();
    private pcs: any;

    private chartMulineStyle = {
        width: '100%',
        height: 392,
        margin: '0 auto',
        padding: '0 15 10 15'
    };

    constructor(props: ParaProps) {
        super(props);
        this.state = {
            // paraSource: [],
            // option: this.hyperParaPic,
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
            max: 1,
            sutrialCount: 10000000,
            succeedRenderCount: 10000000,
            clickCounts: 1,
            isLoadConfirm: false,
            swapyAxis: [],
            paraYdataNested: [],
            isNested: false,
            showFinalMetricKey: "default",
            metricType: 'numberType'
        };
    }

    hyperParaPic = (source: Array<TableObj>, searchSpace: string): void => {
        // filter succeed trials [{}, {}, {}]
        const dataSource = source.filter(filterByStatus);
        const accPara: number[] = [];
        // specific value array
        const eachTrialParams: Array<any> = [];
        // experiment interface search space obj
        const searchRange = searchSpace !== undefined ? JSON.parse(searchSpace) : '';
        // nest search space
        const isNested: boolean = isSearchSpaceNested(searchSpace);
        this.setState({ isNested: isNested });

        // eslint-disable-next-line no-console
        console.log(dataSource);
        // eslint-disable-next-line no-console
        console.log(searchSpace);
        const convertedDataSource = dataSource.map((s) => {
            const ret = { ...(s.description.parameters), ...(s.acc) } as any;
            delete ret.pretrained;
            return ret;
        });

        if (this.pcs === undefined) {
            this.pcs = ParCoords()(this.paraRef.current)
                .data(convertedDataSource)
                // .smoothness(0.15)
                .showControlPoints(false)
                .render()
                .brushMode("1D-axes")
                .reorderable()
                .interactive();
        }
    }

    // get percent value number
    // percentNum = (value: string) => {
    percentNum = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        // percentNum = (event: React.FormEvent<HTMLDivElement>, item?: ISelectableOption) => {
        if (item !== undefined) {
            const vals = parseFloat(item !== undefined ? item.text : '');
            this.setState({ percent: vals / 100, selectedItem: item }, () => {
                this.reInit();
            });
        }
    }
    reInit = (): void => {
        const { dataSource, expSearchSpace } = this.props;
        this.hyperParaPic(dataSource, expSearchSpace);
    }

    // select all final keys
    updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            this.setState({ showFinalMetricKey: item.key }, () => { this.reInit() });
        }
    }

    componentDidMount(): void {
        this.reInit();
    }

    componentDidUpdate(prevProps: ParaProps): void {
        if (this.props.dataSource !== prevProps.dataSource) {
            const { dataSource, expSearchSpace, whichGraph } = this.props;
            if (whichGraph === 'Hyper-parameter') {
                this.hyperParaPic(dataSource, expSearchSpace);
            }
        }
    }

    render(): React.ReactNode {
        const { option, paraNodata, dimName, isLoadConfirm, selectedItem, swapyAxis } = this.state;

        return (
            <div className="parameter">
                <Stack horizontal className="para-filter" horizontalAlign="end">
                    <span className="para-filter-text">Top</span>
                    <Dropdown
                        selectedKey={selectedItem ? selectedItem.key : undefined}
                        onChange={this.percentNum}
                        placeholder="100%"
                        defaultSelectedKeys={[0.2]}
                        options={[
                            { key: '0.2', text: '20%' },
                            { key: '0.5', text: '50%' },
                            { key: '0.8', text: '80%' },
                            { key: '1', text: '100%' },
                        ]}
                        styles={{ dropdown: { width: 120 } }}
                        className="para-filter-percent"
                    />
                    {this.finalKeysDropdown()}
                    <PrimaryButton
                        text="Confirm"
                        disabled={isLoadConfirm}
                    />
                </Stack>
                <div className="parcoords" style={this.chartMulineStyle} ref={this.paraRef} />
            </div>
        );
    }

    private finalKeysDropdown = (): any => {
        const { showFinalMetricKey } = this.state;
        if (TRIALS.finalKeys().length === 1) {
            return null;
        } else {
            const finalKeysDropdown: any = [];
            TRIALS.finalKeys().forEach(item => {
                finalKeysDropdown.push({
                    key: item, text: item
                });
            });
            return (
                <div>
                    <span className="para-filter-text para-filter-middle">Metrics</span>
                    <Dropdown
                        selectedKey={showFinalMetricKey}
                        options={finalKeysDropdown}
                        onChange={this.updateEntries}
                        styles={{ root: { width: 150, display: 'inline-block' } }}
                        className="para-filter-percent"
                    />
                </div>
            );
        }
    };

}

export default Para;
