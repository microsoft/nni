import * as React from 'react';
import axios from 'axios';
import { Stack, StackItem, Panel, PrimaryButton, DefaultButton, Pivot, PivotItem } from '@fluentui/react';
import { infoIcon } from '../../fluent/Icon';
import { DOWNLOAD_IP } from '../../../static/const';
import { downFile } from '../../../static/function';
import MonacoHTML from '../../common/MonacoEditor';
import '../../../static/style/logPanel.scss';

interface LogPanelProps {
    closePanel: () => void;
    activeTab?: string;
}

interface LogPanelState {
    nniManagerLogStr: string | null;
    dispatcherLogStr: string | null;
    isLoading: boolean;
    logPanelHeight: number;
}

class LogPanel extends React.Component<LogPanelProps, LogPanelState> {
    private timerId: number | undefined;

    constructor(props: LogPanelProps) {
        super(props);

        this.state = {
            nniManagerLogStr: null,
            dispatcherLogStr: null,
            isLoading: true,
            logPanelHeight: window.innerHeight
        };
    }

    downloadNNImanager = (): void => {
        if (this.state.nniManagerLogStr !== null) {
            downFile(this.state.nniManagerLogStr, 'nnimanager.log');
        }
    };

    downloadDispatcher = (): void => {
        if (this.state.dispatcherLogStr !== null) {
            downFile(this.state.dispatcherLogStr, 'dispatcher.log');
        }
    };

    dispatcherHTML = (): React.ReactNode => (
        <div>
            <span>Dispatcher log</span>
            <span className='refresh' onClick={this.manualRefresh}>
                {infoIcon}
            </span>
        </div>
    );

    nnimanagerHTML = (): React.ReactNode => (
        <div>
            <span>NNImanager log</span>
            <span className='refresh' onClick={this.manualRefresh}>
                {infoIcon}
            </span>
        </div>
    );

    setLogPanelHeight = (): void => {
        this.setState(() => ({ logPanelHeight: window.innerHeight }));
    };

    async componentDidMount(): Promise<void> {
        this.refresh();
        window.addEventListener('resize', this.setLogPanelHeight);
    }

    componentWillUnmount(): void {
        window.clearTimeout(this.timerId);
        window.removeEventListener('resize', this.setLogPanelHeight);
    }

    render(): React.ReactNode {
        const { closePanel, activeTab } = this.props;
        const { nniManagerLogStr, dispatcherLogStr, isLoading, logPanelHeight } = this.state;
        // tab[height: 56] + tab[margin-bottom: 20] + button[32] + button[margin-top: 45, -bottom: 7] + fluent-panel own paddingBottom[20] + title-border[2]
        const monacoHeight = logPanelHeight - 182;
        return (
            <Stack>
                <Panel
                    isOpen={true}
                    hasCloseButton={false}
                    isFooterAtBottom={true}
                    isLightDismiss={true}
                    onLightDismissClick={closePanel}
                    className='logPanel'
                >
                    <Pivot selectedKey={activeTab} style={{ minHeight: 190 }}>
                        <PivotItem headerText='Dispatcher log' key='dispatcher'>
                            <div className='panel logMargin'>
                                <MonacoHTML
                                    content={dispatcherLogStr || 'Loading...'}
                                    loading={isLoading}
                                    height={monacoHeight}
                                />
                                <Stack horizontal className='buttons'>
                                    <StackItem grow={12} className='download'>
                                        <PrimaryButton text='Download' onClick={this.downloadDispatcher} />
                                    </StackItem>
                                    <StackItem grow={12} className='close'>
                                        <DefaultButton text='Close' onClick={closePanel} />
                                    </StackItem>
                                </Stack>
                            </div>
                        </PivotItem>
                        <PivotItem headerText='NNIManager log' key='nnimanager'>
                            <div className='panel logMargin'>
                                <MonacoHTML
                                    content={nniManagerLogStr || 'Loading...'}
                                    loading={isLoading}
                                    height={monacoHeight}
                                />
                                <Stack horizontal className='buttons'>
                                    <StackItem grow={12} className='download'>
                                        <PrimaryButton text='Download' onClick={this.downloadNNImanager} />
                                    </StackItem>
                                    <StackItem grow={12} className='close'>
                                        <DefaultButton text='Close' onClick={closePanel} />
                                    </StackItem>
                                </Stack>
                            </div>
                        </PivotItem>
                    </Pivot>
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
    };

    private manualRefresh = (): void => {
        this.setState({ isLoading: true });
        this.refresh();
    };
}

export default LogPanel;
