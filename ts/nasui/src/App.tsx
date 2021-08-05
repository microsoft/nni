import React, { ChangeEvent } from 'react';
import './App.css';
import 'typeface-roboto';
import { createStyles, withStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import IconButton from '@material-ui/core/IconButton';
import Slider from '@material-ui/core/Slider';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import RefreshIcon from '@material-ui/icons/Refresh';
import SettingsIcon from '@material-ui/icons/Settings';
import CloseIcon from '@material-ui/icons/Close';
import ShuffleIcon from '@material-ui/icons/Shuffle';
import Snackbar from '@material-ui/core/Snackbar';
import FormControl from '@material-ui/core/FormControl';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormGroup from '@material-ui/core/FormGroup';
import Checkbox from '@material-ui/core/Checkbox';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogContent from '@material-ui/core/DialogContent';
import DialogActions from '@material-ui/core/DialogActions';
import MuiExpansionPanel from '@material-ui/core/ExpansionPanel';
import MuiExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary';
import MuiExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import Backdrop from '@material-ui/core/Backdrop';
import Tooltip from '@material-ui/core/Tooltip';
import Chart from './Chart';
import { Graph } from './graphUtils';

const styles = createStyles({
  bottomAppBar: {
    top: 'auto',
    bottom: 0,
    zIndex: 'auto',
  },
  title: {
    flexGrow: 1,
    textAlign: 'left'
  },
  panel: {
    position: 'absolute',
    top: 76,
    right: 16,
    width: 400,
  },
  listItem: {
    paddingLeft: 0,
    paddingRight: 0,
    paddingTop: 2,
    paddingBottom: 2,
    fontSize: '0.8em',
    wordBreak: 'break-all',
  },
  listSubtitle: {
    fontWeight: 600,
    paddingLeft: 0,
    paddingRight: 0,
    fontSize: '0.9em',
  },
  listTitle: {
    lineHeight: 1.1,
    wordBreak: 'break-all'
  },
  backdrop: {
    color: '#fff',
    zIndex: 100,
  },
  snackbar: {
    bottom: 76
  }
});

const ExpansionPanel = withStyles({
  root: {
    '&$expanded': {
      margin: 'auto',
    },
  },
  expanded: {},
})(MuiExpansionPanel);

const ExpansionPanelSummary = withStyles({
  root: {},
  content: {
    '&$expanded': {
      margin: '12px 0',
    },
  },
  expanded: {},
})(MuiExpansionPanelSummary);

const ExpansionPanelDetails = withStyles(theme => ({
  root: {
    paddingTop: 0,
    paddingBottom: theme.spacing(1),
  },
}))(MuiExpansionPanelDetails);

type AppState = {
  graph: Graph | undefined,
  graphData: any,
  logData: any[],
  sliderValue: number,
  maxSliderValue: number,
  sliderStep: number,
  settingsOpen: boolean,
  hideSidechainNodes: boolean,
  hidePrimitiveNodes: boolean,
  snackbarOpen: boolean,
  selectedNode: string,
  loading: boolean,
  layout: boolean,
}

type AppProps = {
  classes: any
}

class App extends React.Component<AppProps, AppState>  {
  constructor(props: any) {
    super(props);
    this.state = {
      graph: undefined,
      graphData: undefined,
      logData: [],
      sliderValue: 0,
      maxSliderValue: 0,
      sliderStep: 1,
      settingsOpen: false,
      hideSidechainNodes: true,
      hidePrimitiveNodes: true,
      selectedNode: '',
      loading: false,
      snackbarOpen: false,
      layout: false,
    };
    this.refresh = this.refresh.bind(this);
  }

  componentDidMount() {
    this.refresh();
  }

  refresh() {
    this.setState({ loading: true });
    fetch('/refresh')
      .then((response) => { return response.json() })
      .then((data) => {
        const graph = new Graph(data.graph, this.state.hideSidechainNodes);
        this.setState({
          graphData: data.graph,
          graph: graph,
          logData: data.log,
          maxSliderValue: data.log.length - 1,
          sliderStep: Math.max(1, Math.floor(data.log.length / 20)),
          sliderValue: Math.min(data.log.length, this.state.sliderValue),
          loading: false,
          snackbarOpen: graph.nodes.length > 100
        });
      });
  }

  private renderExpansionPanel() {
    const { classes } = this.props;
    const { selectedNode, graph } = this.state;
    if (graph === undefined)
      return null;
    const info = graph.nodeSummary(selectedNode);
    if (info === undefined)
      return null;
    const subtitle = info.op ?
      (info.op === 'IO Node' ? info.op : `Operation: ${info.op}`) :
      `Subgraph: ${info.nodeCount} nodes, ${info.edgeCount} edges`;
    return (
      <ExpansionPanel className={classes.panel}>
        <ExpansionPanelSummary
          expandIcon={<ExpandMoreIcon />}
        >
          <Typography variant='subtitle1' className={classes.listTitle}><b>{info.name}</b><br />{subtitle}</Typography>
        </ExpansionPanelSummary>
        <ExpansionPanelDetails>
          <List dense={true} style={{
            maxHeight: window.innerHeight * .5,
            overflowY: 'auto',
            paddingTop: 0,
            width: '100%'
          }}>
            {
              info.attributes &&
              <React.Fragment>
                <ListItem className={classes.listSubtitle}>Attributes</ListItem>
                <ListItem className={classes.listItem}>{info.attributes}</ListItem>
              </React.Fragment>
            }
            {
              info.inputs.length > 0 &&
              <React.Fragment>
                <ListItem className={classes.listSubtitle}>Inputs ({info.inputs.length})</ListItem>
                {
                  info.inputs.map((item, i) => <ListItem className={classes.listItem} key={`input${i}`}>{item}</ListItem>)
                }
              </React.Fragment>
            }
            {
              info.outputs.length > 0 &&
              <React.Fragment>
                <ListItem className={classes.listSubtitle}>Outputs ({info.outputs.length})</ListItem>
                {
                  info.outputs.map((item, i) => <ListItem className={classes.listItem} key={`output${i}`}>{item}</ListItem>)
                }
              </React.Fragment>
            }
          </List>
        </ExpansionPanelDetails>
      </ExpansionPanel>
    );
  }

  render() {
    const { classes } = this.props;
    const { sliderValue, maxSliderValue, sliderStep, settingsOpen, loading, snackbarOpen } = this.state;
    const handleSliderChange = (event: ChangeEvent<{}>, value: number | number[]) => {
      this.setState({ sliderValue: value as number });
    };
    const handleSettingsDialogToggle = (value: boolean) => () => {
      this.setState({ settingsOpen: value });
    };
    const handleSettingsChange = (name: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
      this.setState({
        ...this.state,
        [name]: event.target.checked
      }, () => {
        this.setState({
          graph: new Graph(this.state.graphData, this.state.hideSidechainNodes),
        })
      });
    };
    const handleSelectionChange = (node: string) => {
      this.setState({
        selectedNode: node
      });
    };
    const handleLoadingState = (state: boolean) => () => {
      this.setState({ loading: state });
    };
    const handleSnackbarClose = () => {
      this.setState({ snackbarOpen: false });
    };
    const handleLayoutStateChanged = (state: boolean) => () => {
      this.setState({ layout: state });
    };
    return (
      <div className='App'>
        <Chart
          width={window.innerWidth}
          height={window.innerHeight}
          graph={this.state.graph}
          activation={sliderValue < this.state.logData.length ? this.state.logData[sliderValue] : undefined}
          handleSelectionChange={handleSelectionChange}
          onRefresh={handleLoadingState(true)}
          onRefreshComplete={handleLoadingState(false)}
          layout={this.state.layout}
          onLayoutComplete={handleLayoutStateChanged(false)}
        />
        <AppBar position='fixed' color='primary'>
          <Toolbar>
            <Typography variant='h6' className={classes.title}>
              NNI NAS Board
            </Typography>
            <Tooltip title="Re-layout graph">
              <IconButton color='inherit' onClick={handleLayoutStateChanged(true)}>
                <ShuffleIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Refresh">
              <IconButton color='inherit' onClick={this.refresh}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Settings">
              <IconButton color='inherit' onClick={handleSettingsDialogToggle(true)}>
                <SettingsIcon />
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>
        <AppBar position='fixed' color='default' className={classes.bottomAppBar}>
          <Toolbar variant='dense'>
            <Grid container spacing={2} alignItems='center'>
              <Grid item xs>
                <Slider
                  value={sliderValue}
                  max={maxSliderValue}
                  min={0}
                  step={sliderStep}
                  onChange={handleSliderChange}
                />
              </Grid>
              <Grid item>
                <Typography variant='body1'>
                  {sliderValue}/{maxSliderValue}
                </Typography>
              </Grid>
            </Grid>
          </Toolbar>
        </AppBar>
        <Dialog onClose={handleSettingsDialogToggle(false)} open={settingsOpen}>
          <DialogTitle>Settings</DialogTitle>
          <DialogContent>
            <FormControl component='fieldset'>
              <FormGroup>
                <FormControlLabel
                  control={<Checkbox checked={this.state.hideSidechainNodes}
                    onChange={handleSettingsChange('hideSidechainNodes')}
                    value='hideSidechainNodes' />}
                  label='Hide sidechain nodes'
                />
                { // TODO: hide primitive nodes
                /* <FormControlLabel
                  control={<Checkbox checked={this.state.hidePrimitiveNodes}
                    onChange={handleSettingsChange('hidePrimitiveNodes')}
                    value='hidePrimitiveNodes' />}
                  label='Hide primitive nodes'
                /> */}
              </FormGroup>
            </FormControl>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleSettingsDialogToggle(false)} color='primary'>
              Close
            </Button>
          </DialogActions>
        </Dialog>
        {this.renderExpansionPanel()}
        <Snackbar
          className={classes.snackbar}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'left',
          }}
          open={snackbarOpen}
          message='Graph is too large. Might induce performance issue.'
          onClose={handleSnackbarClose}
          action={
            <IconButton size='small' color='inherit' onClick={handleSnackbarClose}>
              <CloseIcon fontSize='small' />
            </IconButton>
          }
        />
        {
          loading && <Backdrop className={classes.backdrop} open={true}>
            <Typography>Loading...</Typography>
          </Backdrop>
        }
      </div>
    );
  }
}

export default withStyles(styles)(App);
