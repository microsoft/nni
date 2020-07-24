import * as React from 'react';
import axios from 'axios';
import {
    Stack, StackItem, Panel, PrimaryButton, DefaultButton,
    Pivot, PivotItem
} from 'office-ui-fabric-react';
import { infoIcon } from '../Buttons/Icon';
import { DOWNLOAD_IP } from '../../static/const';
import { downFile } from '../../static/function';
import MonacoHTML from '../public-child/MonacoEditor';
import '../../static/style/logDrawer.scss';

interface LogDrawerProps {
    closeDrawer: () => void;
    activeTab?: string;
}

interface LogDrawerState {
    nniManagerLogStr: string | null;
    dispatcherLogStr: string | null;
    isLoading: boolean;
    logDrawerHeight: number;
}

class LogDrawer extends React.Component<LogDrawerProps, LogDrawerState> {
    private timerId: number | undefined;

    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            nniManagerLogStr: null,
            dispatcherLogStr: null,
            isLoading: true,
            logDrawerHeight: window.innerHeight - 48
        };
    }

    downloadNNImanager = (): void => {
        if (this.state.nniManagerLogStr !== null) {
            downFile(this.state.nniManagerLogStr, 'nnimanager.log');
        }
    }

    downloadDispatcher = (): void => {
        if (this.state.dispatcherLogStr !== null) {
            downFile(this.state.dispatcherLogStr, 'dispatcher.log');
        }
    }

    dispatcherHTML = (): React.ReactNode => {
        return (
            <div>
                <span>Dispatcher Log</span>
                <span className="refresh" onClick={this.manualRefresh}>
                    {infoIcon}
                </span>
            </div>
        );
    }

    nnimanagerHTML = (): React.ReactNode => {
        return (
            <div>
                <span>NNImanager Log</span>
                <span className="refresh" onClick={this.manualRefresh}>{infoIcon}</span>
            </div>
        );
    }

    setLogDrawerHeight = (): void => {
        this.setState(() => ({ logDrawerHeight: window.innerHeight - 48 }));
    }

    async componentDidMount(): Promise<void> {
        this.refresh();
        window.addEventListener('resize', this.setLogDrawerHeight);
    }

    componentWillUnmount(): void{
        window.clearTimeout(this.timerId);
        window.removeEventListener('resize', this.setLogDrawerHeight);
    }

    render(): React.ReactNode {
        const { closeDrawer, activeTab } = this.props;
        const { nniManagerLogStr, dispatcherLogStr, isLoading, logDrawerHeight } = this.state;

        return (
            <Stack>
                <Panel
                    isOpen={true}
                    hasCloseButton={false}
                    isFooterAtBottom={true}
                    isLightDismiss={true}
                    onLightDismissClick={closeDrawer}
                >
                    <div className="log-tab-body">
                        <Pivot
                            selectedKey={activeTab}
                            style={{ minHeight: 190, paddingTop: '16px' }}
                        >
                            {/* <PivotItem headerText={this.dispatcherHTML()} key="dispatcher" onLinkClick> */}
                            <PivotItem headerText="Dispatcher Log" key="dispatcher">
                                <MonacoHTML
                                    content={dispatcherLogStr || 'Loading...'}
                                    loading={isLoading}
                                    // paddingTop[16] + tab[44] + button[32]
                                    height={logDrawerHeight - 92}
                                />
                                <Stack horizontal className="buttons">
                                    <StackItem grow={12} className="download">
                                        <PrimaryButton text="Download" onClick={this.downloadDispatcher} />
                                    </StackItem>
                                    <StackItem grow={12} className="close">
                                        <DefaultButton text="Close" onClick={closeDrawer} />
                                    </StackItem>
                                </Stack>
                            </PivotItem>
                            <PivotItem headerText="NNIManager Log" key="nnimanager">
                                {/* <TabPane tab="NNImanager Log" key="nnimanager"> */}
                                <MonacoHTML
                                    content={nniManagerLogStr || 'Loading...'}
                                    loading={isLoading}
                                    height={logDrawerHeight - 92}
                                />
                                <Stack horizontal className="buttons">
                                    <StackItem grow={12} className="download">
                                        <PrimaryButton text="Download" onClick={this.downloadNNImanager} />
                                    </StackItem>
                                    <StackItem grow={12} className="close">
                                        <DefaultButton text="Close" onClick={closeDrawer} />
                                    </StackItem>
                                </Stack>
                            </PivotItem>
                        </Pivot>
                    </div>
                </Panel>
            </Stack>
        );
    }

    private refresh = (): void => {
        window.clearTimeout(this.timerId);
        const dispatcherPromise = axios.get(`${DOWNLOAD_IP}/dispatcher.log`);
        const nniManagerPromise = axios.get(`${DOWNLOAD_IP}/nnimanager.log`);
        dispatcherPromise.then(res => {
            if (res.status === 200) {
                this.setState({ dispatcherLogStr: res.data });
            }
        });
        nniManagerPromise.then(res => {
            if (res.status === 200) {
                this.setState({ nniManagerLogStr: res.data });
            }
        });
        Promise.all([dispatcherPromise, nniManagerPromise]).then(() => {
            this.setState({ isLoading: false });
            this.timerId = window.setTimeout(this.refresh, 10000);
        });
    }

    private manualRefresh = (): void => {
        this.setState({ isLoading: true });
        this.refresh();
    }
}

export default LogDrawer;
