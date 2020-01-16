import * as React from 'react';
import { WEBUIDOC } from '../static/const';
// import { verticalAlign } from '../static/style/commonSty';
import {
    Stack, initializeIcons, IconButton,
    StackItem, IContextualMenuProps } from 'office-ui-fabric-react'; // eslint-disable-line no-unused-vars
// import MediaQuery from 'react-responsive';
import LogDrawer from './Modal/LogDrawer';
import ExperimentDrawer from './Modal/ExperimentDrawer';
import { OVERVIEWTABS, DETAILTABS, NNILOGO } from './stateless-component/NNItabs';
import '../static/style/nav/nav.scss';
import '../static/style/icon.scss';

// 初始化icon
initializeIcons();

interface NavState {
    version: string;
    menuVisible: boolean;
    navBarVisible: boolean;
    isdisabledFresh: boolean;
    isvisibleLogDrawer: boolean;
    isvisibleExperimentDrawer: boolean;
    activeKey: string;
}

class NavCon extends React.Component<{}, NavState> {

    constructor(props: {}) {
        super(props);
        this.state = {
            version: '',
            menuVisible: false,
            navBarVisible: false,
            isdisabledFresh: false,
            isvisibleLogDrawer: false, // download button (nnimanager·dispatcher) click -> drawer
            isvisibleExperimentDrawer: false,
            activeKey: 'dispatcher'
        };
    }

    // to see & download experiment parameters
    showExpcontent = (): void => {
        this.setState({ isvisibleExperimentDrawer: true });
    }
    // to see & download nnimanager log
    showNNImanagerLog = (): void => {
        this.setState({ activeKey: 'nnimanager', isvisibleLogDrawer: true });
    }
    // to see & download dispatcher log
    showDispatcherLog = (): void => {
        this.setState({ isvisibleLogDrawer: true, activeKey: 'dispatcher' });
    }

    // refresh current page
    fresh = (event: React.SyntheticEvent<EventTarget>): void => {
        event.preventDefault();
        event.stopPropagation();
        this.setState({ isdisabledFresh: true }, () => {
            setTimeout(() => { this.setState({ isdisabledFresh: false }); }, 1000);
        });
    }

    // close log drawer (nnimanager.dispatcher)
    closeLogDrawer = (): void => {
        this.setState({ isvisibleLogDrawer: false, activeKey: '' });
    }

    // close download experiment parameters drawer
    closeExpDrawer = (): void => {
        this.setState({ isvisibleExperimentDrawer: false });
    }

    render(): React.ReactNode {
        const { isvisibleLogDrawer, activeKey, isvisibleExperimentDrawer, version } = this.state;
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Stack horizontal className="nav">
                <StackItem grow={30} styles={{ root: { minWidth: 300, display: 'flex', verticalAlign: 'center' } }}>
                    <span className="desktop-logo">{NNILOGO}</span>
                    <span className="left-right-margin">{OVERVIEWTABS}</span>
                    <span>{DETAILTABS}</span>
                </StackItem>
                <StackItem grow={70} className="veralign">
                    {/* TODO: min width 根据实际的最小宽度来定 */}
                    <Stack horizontal horizontalAlign="end" gap={30} styles={{ root: { minWidth: 400, color: '#fff' } }}>
                        {/* refresh button */}
                        <Stack.Item align="center">
                            <IconButton
                                className="iconButtons"
                                iconProps={{ iconName: 'sync' }}
                                title="refresh"
                                ariaLabel="refresh"
                                // disabled={true}
                                onClick={this.fresh}
                            />
                        </Stack.Item>

                        {/* <StackItem>
                            refresh selector
                        </StackItem> */}
                        {/* view button download log*/}
                        <IconButton
                            className="iconButtons"
                            menuProps={this.menuProps}
                            iconProps={{ iconName: 'View' }}
                            title="view"
                            ariaLabel="view"
                        // onMenuClick={this.iconbuttonss}
                        // onMenuClick?: (ev?: React.MouseEvent<HTMLElement> | React.KeyboardEvent<HTMLElement>, button?: IButtonProps) => void;
                        />
                        {/* link to document button */}
                        <a href={WEBUIDOC} target="_blank" rel="noopener noreferrer" className="docIcon">
                            <IconButton
                                className="iconButtons"
                                iconProps={{ iconName: 'StatusCircleQuestionMark' }}
                                title="document"
                                ariaLabel="document"
                            />
                        </a>
                        {/* <a href= target="_blank"> */}
                        <a href={feed} target="_blank" rel="noopener noreferrer" className="feedback">
                            <IconButton
                                className="iconButtons"
                                iconProps={{ iconName: 'OfficeChat' }}
                                title="feedback"
                                ariaLabel="feedback"
                            />
                        </a>
                        {/* <span className="version">Version: {version}</span> */}
                        <span>v1.3</span>
                    </Stack>
                </StackItem>
                {/* the drawer for dispatcher & nnimanager log message */}
                {isvisibleLogDrawer ? (
                    <LogDrawer
                        closeDrawer={this.closeLogDrawer}
                        activeTab={activeKey}
                    />
                ) : null}
                <ExperimentDrawer
                    isVisble={isvisibleExperimentDrawer}
                    closeExpDrawer={this.closeExpDrawer}
                />
            </Stack>
        );
    }

    // view and download experiment [log & experiment result]
    private menuProps: IContextualMenuProps = {
        items: [
            {
                key: 'experiment',
                text: 'Experiment Parameters',
                iconProps: { iconName: 'Mail' },
                onClick: this.showExpcontent
            },
            {
                key: 'managerlog',
                text: 'NNImanager Logfile',
                iconProps: { iconName: 'Calendar' },
                onClick: this.showNNImanagerLog
            },
            {
                key: 'dispatcherlog',
                text: 'Dispatcher Logfile',
                iconProps: { iconName: 'Calendar' },
                onClick: this.showDispatcherLog
            }
        ],
        directionalHintFixed: true
    };
}

export default NavCon;
