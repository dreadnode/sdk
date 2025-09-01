#!/usr/bin/env python
# Requirements - pip install neo4j-driver
# This script is used to create randomized sample databases.
# Commands
# 	dbconfig - Set the credentials and URL for the database you're connecting too
# 	connect - Connects to the database using supplied credentials
# 	setparams - Set the settings JSON file
# 	setdomain - Set the domain name
# 	cleardb - Clears the database and sets the schema properly
# 	generate - Connects to the database, clears the DB, sets the schema, and generates random data

# import cmd
import math
import random
import time
import uuid

import fire
from adsimulator.generators.acls import (
    generate_administrators_acls,
    generate_all_extended_rights,
    generate_default_dc_groups_acls,
    generate_default_groups_acls,
    generate_default_users_acls,
    generate_domain_admins_acls,
    generate_enterprise_admins_acls,
    generate_generic_all,
    generate_generic_write,
    generate_local_admin_rights,
    generate_outbound_acls,
    generate_owns,
    generate_write_dacl,
    generate_write_owner,
)
from adsimulator.generators.computers import (
    generate_allowed_to_delegate_relationships_on_computers,
    generate_allowed_to_delegate_relationships_on_it_users,
    generate_can_ps_remote_relationships_on_it_groups,
    generate_can_ps_remote_relationships_on_it_users,
    generate_can_rdp_relationships_on_it_groups,
    generate_can_rdp_relationships_on_it_users,
    generate_computers,
    generate_dcom_relationships_on_it_groups,
    generate_dcom_relationships_on_it_users,
    generate_dcs,
    generate_default_admin_to,
    generate_sessions,
)
from adsimulator.generators.domains import generate_domain, generate_trusts
from adsimulator.generators.gpos import (
    generate_default_gpos,
    generate_gpos,
    gplink_domain_to_ous,
    link_default_gpos,
    link_gpos_to_ous,
)
from adsimulator.generators.groups import (
    assign_users_to_group,
    generate_default_groups,
    generate_default_member_of,
    generate_domain_administrators,
    generate_groups,
    nest_groups,
)
from adsimulator.generators.ous import (
    generate_computer_ous,
    generate_domain_controllers_ou,
    generate_user_ous,
    link_ous_to_domain,
)
from adsimulator.generators.users import (
    generate_administrator,
    generate_default_account,
    generate_guest_user,
    generate_krbtgt_user,
    generate_users,
    link_default_users_to_domain,
)
from adsimulator.templates.default_values import DEFAULT_VALUES
from adsimulator.utils.data import (
    get_domains_pool,
    get_names_pool,
    get_parameters_from_json,
    get_surnames_pool,
)
from adsimulator.utils.domains import get_domain_dn
from adsimulator.utils.parameters import (
    get_int_param_value,
    get_int_param_value_with_upper_limit,
    get_perc_param_value,
    print_all_parameters,
)
from neo4j import GraphDatabase


class Messages:
    def title(self):
        print("==================================================================")
        print(
            """

      ,.,
      MMMM_    ,..,
        \"_ \"__"MMMMM          ,...,,
 ,..., __.\" --\"    ,.,     _-\"MMMMMMM
MMMMMM"___ "_._   MMM"_."" _ """
            """         _     _                 _       _               
 \"\"\"\"\"    \"\" , \\_.   \"_. .\"  __ _  __| |___(_)_ __ ___  _   _| | __ _| |_ ___  _ __    
        ,., _"__ \\__./ ."   / _` |/ _` / __| | '_ ` _ \\| | | | |/ _` | __/ _ \\| '__|   
       MMMMM_"  "_    ./   | (_| | (_| \\__ \\ | | | | | | |_| | | (_| | || (_) | | 
        ''''      (    )    \\__,_|\\__,_|___/_|_| |_| |_|\\__,_|_|\\__,_|\\__\\___/|_|  
 ._______________.-'____\"---._.
  \\                          /
   \\________________________/
   (_)                    (_)

                                                         
                                                          
        """
        )
        print(" A realistic simulator of Active Directory domains\n")
        print("==================================================================")

    def input_default(self, prompt, default):
        return input("%s [%s] " % (prompt, default)) or default

    def input_yesno(self, prompt, default):
        temp = input(
            prompt + " " + ("Y" if default else "y") + "/" + ("n" if default else "N") + " "
        )
        if temp == "y" or temp == "Y":
            return True
        if temp == "n" or temp == "N":
            return False
        return default


# class MainMenu(cmd.Cmd):
class MainMenu:
    def __init__(
        self,
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        domain="SKY.NET",
        #        save=False,
        parameters=DEFAULT_VALUES,
    ):
        self.m = Messages()
        self.url = url
        self.username = username
        self.password = password
        self.use_encryption = False
        self.driver = None
        self.connected = False
        self.old_domain = None
        self.domain = domain
        self.current_time = int(time.time())
        self.base_sid = "S-1-5-21-883232822-274137685-4173207997"
        self.first_names = get_names_pool()
        self.last_names = get_surnames_pool()
        self.domain_names = get_domains_pool()
        self.parameters_json_path = "DEFAULT"
        self.parameters = parameters
        self.json_file_name = None

    #        self.save = save

    def dbconfig(self, args):
        print("Current Settings:")
        print(f"DB Url: {self.url}")
        print(f"DB Username: {self.username}")
        print(f"DB Password: {self.password}")
        print(f"Use encryption: {self.use_encryption}")
        print()
        self.url = self.m.input_default("Enter DB URL", self.url)
        self.username = self.m.input_default("Enter DB Username", self.username)
        self.password = self.m.input_default("Enter DB Password", self.password)

        self.use_encryption = self.m.input_yesno("Use encryption?", self.use_encryption)
        print()
        print("New Settings:")
        print(f"DB Url: {self.url}")
        print(f"DB Username: {self.username}")
        print(f"DB Password: {self.password}")
        print(f"Use encryption: {self.use_encryption}")
        print()
        print("Testing DB Connection")
        self.test_db_conn()

    def setdomain(self, args):
        passed = args
        if passed != "":
            try:
                self.domain = passed.upper()
                return
            except ValueError:
                pass

        self.domain = self.m.input_default("Domain", self.domain).upper()
        print()
        print("New Settings:")
        print(f"Domain: {self.domain}")

    def exit(self):
        raise KeyboardInterrupt

    def connect(self):
        self.test_db_conn()

    def cleardb(self):
        if not self.connected:
            print("Not connected to database. Use connect first")
            return

        print("Clearing Database")
        d = self.driver
        session = d.session()

        session.run("match (a) -[r] -> () delete a, r")
        session.run("match (a) delete a")

        session.close()

        print("DB Cleared and Schema Set")

    def setparams(self, args=""):
        passed = args
        if passed != "":
            try:
                json_path = passed
                self.parameters = get_parameters_from_json(json_path)
                self.parameters_json_path = json_path
                print_all_parameters(self.parameters)
                return
            except ValueError:
                pass

        json_path = self.m.input_default("Parameters JSON file", self.parameters_json_path)
        self.parameters = get_parameters_from_json(json_path)
        if self.parameters == DEFAULT_VALUES:
            self.parameters_json_path = "DEFAULT"
        else:
            self.parameters_json_path = json_path
        self.parameters = get_parameters_from_json(json_path)
        print_all_parameters(self.parameters)

    def test_db_conn(self):
        self.connected = False
        if self.driver is not None:
            self.driver.close()
        try:
            self.driver = GraphDatabase.driver(
                self.url,
                auth=(self.username, self.password),
                encrypted=self.use_encryption,
            )
            self.connected = True
            print("Database Connection Successful!")
        except:
            self.connected = False
            print("Database Connection Failed. Check your settings.")

    def generate(self):
        self.test_db_conn()
        self.cleardb()
        self.generate_data()
        self.old_domain = self.domain

    def generate_data(self):
        if not self.connected:
            print("Not connected to database. Use connect first")
            return

        domain_dn = get_domain_dn(self.domain)

        computers = []
        computer_properties_list = []
        dc_properties_list = []
        groups = []
        users = []
        user_properties_list = []
        gpos = []
        gpos_properties_list = []
        ou_guid_map = {}
        ou_properties_list = []

        session = self.driver.session()

        print("Starting data generation")

        print("Generating the", self.domain, "domain")
        functional_level = generate_domain(
            session, self.domain, self.base_sid, domain_dn, self.parameters
        )

        print("Generating the default domain groups")
        generate_default_groups(session, self.domain, self.base_sid, self.old_domain)

        ddp = str(uuid.uuid4()).upper()
        ddcp = str(uuid.uuid4()).upper()
        dcou = str(uuid.uuid4()).upper()

        print("Generating default GPOs")
        generate_default_gpos(session, self.domain, domain_dn, ddp, ddcp)

        print("Generating Domain Controllers OU")
        generate_domain_controllers_ou(session, self.domain, domain_dn, dcou)

        print("Linking Default GPOs")
        link_default_gpos(session, self.domain, ddp, ddcp, dcou)

        print("Generating Enterprise Admins ACLs")
        generate_enterprise_admins_acls(session, self.domain)

        print("Generating Administrators ACLs")
        generate_administrators_acls(session, self.domain)

        print("Generating Domain Admins ACLs")
        generate_domain_admins_acls(session, self.domain)

        print("Generating DC groups ACLs")
        generate_default_dc_groups_acls(session, self.domain)

        num_computers = get_int_param_value("Computer", "nComputers", self.parameters)
        print("Generating", str(num_computers), "computers")
        computer_properties_list, computers, ridcount = generate_computers(
            session,
            self.domain,
            self.base_sid,
            num_computers,
            computers,
            self.current_time,
            self.parameters,
        )

        num_ous = get_int_param_value_with_upper_limit("OU", "nOUs", self.parameters, 50)
        if not num_ous % 2 == 0:
            num_ous = num_ous - 1
        num_states = int(num_ous / 2)

        print("Generating", str(num_states), "Domain Controllers")
        dc_properties_list, ridcount = generate_dcs(
            session,
            self.domain,
            self.base_sid,
            domain_dn,
            num_states,
            dcou,
            ridcount,
            self.current_time,
            self.parameters,
            functional_level,
        )

        print("Generating default users")
        generate_guest_user(session, self.domain, self.base_sid, self.parameters)
        generate_default_account(session, self.domain, self.base_sid, self.parameters)
        generate_administrator(session, self.domain, self.base_sid, self.parameters)
        generate_krbtgt_user(session, self.domain, self.base_sid, self.parameters)
        link_default_users_to_domain(session, self.domain, self.base_sid)

        num_users = get_int_param_value("User", "nUsers", self.parameters)
        print("Generating", str(num_users), "users")
        user_properties_list, users, ridcount = generate_users(
            session,
            self.domain,
            self.base_sid,
            num_users,
            self.current_time,
            self.first_names,
            self.last_names,
            users,
            ridcount,
            self.parameters,
        )

        num_groups = get_int_param_value("Group", "nGroups", self.parameters)
        print("Generating groups")
        group_properties_list, groups, ridcount = generate_groups(
            session,
            self.domain,
            self.base_sid,
            domain_dn,
            num_groups,
            groups,
            ridcount,
            self.parameters,
        )

        print("Adding Domain Admins to Local Admins of Computers")
        generate_default_admin_to(session, self.base_sid)

        das = generate_domain_administrators(session, self.domain, num_users, users)

        print("Adding members to default groups")
        generate_default_member_of(session, self.domain, self.base_sid, self.old_domain)

        nesting_perc = get_perc_param_value("Group", "nestingGroupProbability", self.parameters)
        print(
            "Applying random group nesting (nesting probability:",
            str(nesting_perc),
            "%)",
        )
        nest_groups(session, num_groups, groups, nesting_perc)

        print("Adding users to groups")
        it_users = assign_users_to_group(session, num_users, users, groups, das, self.parameters)

        print("Adding local admin rights")
        it_groups = generate_local_admin_rights(session, groups, computers)

        print("Adding ACLs for default groups")
        generate_default_groups_acls(session, self.domain, self.base_sid)

        print("Adding ACLs for default users")
        generate_default_users_acls(session, self.domain, self.base_sid)

        print("Adding AllExtendedRights")
        generate_all_extended_rights(session, self.domain, self.base_sid, user_properties_list, das)

        can_rdp_users_perc = get_perc_param_value(
            "Computer", "CanRDPFromUserPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (can_rdp_users_perc / 100)))
        print("Adding a maximum of", str(count), "CanRDP from users")
        generate_can_rdp_relationships_on_it_users(session, computers, it_users, count)

        can_rdp_groups_perc = get_perc_param_value(
            "Computer", "CanRDPFromGroupPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (can_rdp_groups_perc / 100)))
        print("Adding a maximum of", str(count), "CanRDP from groups")
        generate_can_rdp_relationships_on_it_groups(session, computers, it_groups, count)

        dcom_users_perc = get_perc_param_value(
            "Computer", "ExecuteDCOMFromUserPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (dcom_users_perc / 100)))
        print("Adding a maximum of", str(count), "ExecuteDCOM from users")
        generate_dcom_relationships_on_it_users(session, computers, it_users, count)

        dcom_groups_perc = get_perc_param_value(
            "Computer", "ExecuteDCOMFromGroupPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (dcom_groups_perc / 100)))
        print("Adding a maximum of", str(count), "ExecuteDCOM from groups")
        generate_dcom_relationships_on_it_groups(session, computers, it_groups, count)

        allowed_to_delegate_users_perc = get_perc_param_value(
            "Computer", "AllowedToDelegateFromUserPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (allowed_to_delegate_users_perc / 100)))
        print("Adding a maximum of", str(count), "AllowedToDelegate from users")
        generate_allowed_to_delegate_relationships_on_it_users(session, computers, it_users, count)

        allowed_to_delegate_computers_perc = get_perc_param_value(
            "Computer", "AllowedToDelegateFromComputerPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (allowed_to_delegate_computers_perc / 100)))
        print("Adding a maximum of", str(count), "AllowedToDelegate from computers")
        generate_allowed_to_delegate_relationships_on_computers(session, computers, count)

        ps_remote_users_perc = get_perc_param_value(
            "Computer", "CanPSRemoteFromUserPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (ps_remote_users_perc / 100)))
        print("Adding a maximum of", str(count), "CanPSRemote from users")
        generate_can_ps_remote_relationships_on_it_users(session, computers, it_users, count)

        ps_remote_groups_perc = get_perc_param_value(
            "Computer", "CanPSRemoteFromGroupPercentage", self.parameters
        )
        count = int(math.floor(len(computers) * (ps_remote_groups_perc / 100)))
        print("Adding a maximum of", str(count), "CanPSRemote from groups")
        generate_can_ps_remote_relationships_on_it_groups(session, computers, it_groups, count)

        print("Adding sessions")
        generate_sessions(session, num_users, computers, users, das)

        print("Generating", str(num_ous), "OUs")
        split_num_computers = int(math.ceil(num_computers / num_states))
        split_num_users = int(math.ceil(num_users / num_states))
        ou_properties_list, ou_guid_map = generate_computer_ous(
            session,
            self.domain,
            domain_dn,
            computers,
            ou_guid_map,
            ou_properties_list,
            split_num_computers,
            num_states,
        )
        ou_properties_list, ou_guid_map = generate_user_ous(
            session,
            self.domain,
            domain_dn,
            users,
            ou_guid_map,
            ou_properties_list,
            split_num_users,
            num_states,
        )
        link_ous_to_domain(session, self.domain, ou_guid_map)

        num_gpos = get_int_param_value_with_upper_limit(
            "GPO", "nGPOs", self.parameters, 2 * num_states
        )
        print("Creating", str(num_gpos), "GPOs")
        gpos, gpos_properties_list = generate_gpos(
            session,
            self.domain,
            domain_dn,
            gpos,
            gpos_properties_list,
            num_gpos,
            self.parameters,
        )

        print("Generating GpLink")
        ou_names = list(ou_guid_map.keys())
        link_gpos_to_ous(session, gpos, ou_names, ou_guid_map)
        gplink_domain_to_ous(session, self.domain, ou_names, ou_guid_map)

        gpos.append(f"DEFAULT DOMAIN POLICY@{self.domain}")
        gpos.append(f"DEFAULT DOMAIN CONTROLLERS POLICY@{self.domain}")

        print("Adding GenericWrite")
        generate_generic_write(
            session,
            computer_properties_list,
            user_properties_list,
            group_properties_list,
            gpos_properties_list,
            das,
            self.domain,
            self.base_sid,
        )
        print("Adding Owns")
        generate_owns(
            session,
            computer_properties_list,
            user_properties_list,
            group_properties_list,
            ou_properties_list,
            gpos_properties_list,
            self.domain,
            self.base_sid,
        )
        print("Adding WriteDacl")
        generate_write_dacl(
            session,
            dcou,
            computer_properties_list,
            user_properties_list,
            group_properties_list,
            ou_properties_list,
            gpos_properties_list,
            das,
            self.domain,
            self.base_sid,
        )
        print("Adding WriteOwner")
        generate_write_owner(
            session,
            dcou,
            computer_properties_list,
            user_properties_list,
            group_properties_list,
            ou_properties_list,
            gpos_properties_list,
            das,
            self.domain,
            self.base_sid,
        )
        print("Adding GenericAll")
        generate_generic_all(
            session,
            dcou,
            dc_properties_list,
            computer_properties_list,
            user_properties_list,
            group_properties_list,
            ou_properties_list,
            gpos_properties_list,
            das,
            self.domain,
            self.base_sid,
        )

        acl_principals_perc = get_perc_param_value(
            "ACLs", "ACLPrincipalsPercentage", self.parameters
        )
        num_acl_principals = int(round(len(it_groups) * (acl_principals_perc / 100)))
        generate_outbound_acls(
            session,
            num_acl_principals,
            it_groups,
            it_users,
            gpos,
            computers,
            self.parameters,
        )

        session.run("MATCH (n) SET n.domain=$domain", domain=self.domain)

        self.domain_names = get_domains_pool()
        generate_trusts(session, self.domain, self.domain_names, self.parameters)

        session.run("MATCH (n:User) SET n.owned=false")

        owned_user = random.choice(users)
        print("Compromised user:", owned_user)

        # session.run('MATCH (n:User {name: $owneduser}) SET n.owned=true', owneduser=owned_user)
        # session.run('MATCH (n:User {name: $owneduser}) SET n:Compromised', owneduser=owned_user)

        session.run("MATCH (n:Computer) SET n.owned=false")

        # if self.save == True:
        #     self.write_json(session)

        session.close()

        print("Database Generation Finished!")

    # def write_json(self, session):
    #     json_filename = uuid.uuid4().hex + ".jsonl"
    #     query = "CALL apoc.export.json.all('" + json_filename + "',{useTypes:true})"

    #     try:
    #         session.run(query)
    #     except Exception as error:
    #         print(error)

    #     print("Graph exported to: /var/lib/neo4j/import/{}".format(json_filename))


def optuna_generate(args):
    env = MainMenu(
        url=args.url,
        username=args.username,
        password=args.password,
        domain=args.domain,
        #        save=args.save,
    )

    try:
        env.connect()
        print("[+] Successfully connected!")
    except Exception as error:
        print(f"[!] Failed to connect! {error}")

    def objective(trial):
        config = DEFAULT_VALUES.copy()
        config["Domain"]["functionalLevelProbability"]["2008"] = trial.suggest_int("2008", 1, 10)
        config["Domain"]["functionalLevelProbability"]["2012"] = trial.suggest_int("2012", 1, 50)
        config["Domain"]["functionalLevelProbability"]["2012 R2"] = trial.suggest_int(
            "2012 R2", 1, 50
        )
        config["Domain"]["functionalLevelProbability"]["2016"] = trial.suggest_int("2016", 1, 50)
        config["Domain"]["functionalLevelProbability"]["Unknown"] = trial.suggest_int(
            "Unknown", 1, 50
        )
        config["Domain"]["Trusts"]["SIDFilteringProbability"] = trial.suggest_int(
            "SIDFilteringProbability", 1, 100
        )
        config["Domain"]["Trusts"]["Inbound"] = trial.suggest_int("Inbound", 1, 100)
        config["Domain"]["Trusts"]["Outbound"] = trial.suggest_int("Outbound", 1, 100)
        config["Domain"]["Trusts"]["Bidirectional"] = trial.suggest_int("Bidirectional", 1, 100)

        # Define your parameters for the 'Computer' section
        config["Computer"]["nComputers"] = trial.suggest_int("nComputers", 50, 1000)
        config["Computer"]["CanRDPFromUserPercentage"] = trial.suggest_int(
            "CanRDPFromUserPercentage", 1, 50
        )
        config["Computer"]["CanRDPFromGroupPercentage"] = trial.suggest_int(
            "CanRDPFromGroupPercentage", 1, 50
        )
        config["Computer"]["CanPSRemoteFromUserPercentage"] = trial.suggest_int(
            "CanPSRemoteFromUserPercentage", 1, 50
        )
        config["Computer"]["CanPSRemoteFromGroupPercentage"] = trial.suggest_int(
            "CanPSRemoteFromGroupPercentage", 1, 50
        )
        config["Computer"]["ExecuteDCOMFromUserPercentage"] = trial.suggest_int(
            "ExecuteDCOMFromUserPercentage", 1, 50
        )
        config["Computer"]["ExecuteDCOMFromGroupPercentage"] = trial.suggest_int(
            "ExecuteDCOMFromGroupPercentage", 1, 50
        )
        config["Computer"]["AllowedToDelegateFromUserPercentage"] = trial.suggest_int(
            "AllowedToDelegateFromUserPercentage", 1, 50
        )
        config["Computer"]["AllowedToDelegateFromComputerPercentage"] = trial.suggest_int(
            "AllowedToDelegateFromComputerPercentage", 1, 50
        )
        config["Computer"]["enabled"] = trial.suggest_int("enabled", 1, 100)
        config["Computer"]["haslaps"] = trial.suggest_int("haslaps", 1, 50)
        config["Computer"]["unconstraineddelegation"] = trial.suggest_int(
            "unconstraineddelegation", 1, 50
        )
        config["Computer"]["privesc"] = trial.suggest_int("privesc", 1, 50)
        config["Computer"]["creddump"] = trial.suggest_int("creddump", 1, 50)
        config["Computer"]["exploitable"] = trial.suggest_int("exploitable", 1, 50)
        config["Computer"]["osProbability"]["Windows XP Professional Service Pack 3"] = (
            trial.suggest_int("Windows XP Professional Service Pack 3", 1, 10)
        )
        config["Computer"]["osProbability"]["Windows 7 Professional Service Pack 1"] = (
            trial.suggest_int("Windows 7 Professional Service Pack 1", 1, 20)
        )
        config["Computer"]["osProbability"]["Windows 7 Ultimate Service Pack 1"] = (
            trial.suggest_int("Windows 7 Ultimate Service Pack 1", 1, 20)
        )
        config["Computer"]["osProbability"]["Windows 7 Enterprise Service Pack 1"] = (
            trial.suggest_int("Windows 7 Enterprise Service Pack 1", 1, 20)
        )
        config["Computer"]["osProbability"]["Windows 10 Pro"] = trial.suggest_int(
            "Windows 10 Pro", 1, 40
        )
        config["Computer"]["osProbability"]["Windows 10 Enterprise"] = trial.suggest_int(
            "Windows 10 Enterprise", 1, 40
        )

        config["DC"]["enabled"] = trial.suggest_int("DC_enabled", 1, 100)
        config["DC"]["haslaps"] = trial.suggest_int("DC_haslaps", 1, 50)
        config["DC"]["osProbability"]["Windows Server 2003 Enterprise Edition"] = trial.suggest_int(
            "DC_Windows Server 2003 Enterprise Edition", 1, 10
        )
        config["DC"]["osProbability"]["Windows Server 2008 Standard"] = trial.suggest_int(
            "DC_Windows Server 2008 Standard", 1, 10
        )
        config["DC"]["osProbability"]["Windows Server 2008 Datacenter"] = trial.suggest_int(
            "DC_Windows Server 2008 Datacenter", 1, 10
        )
        config["DC"]["osProbability"]["Windows Server 2008 Enterprise"] = trial.suggest_int(
            "DC_Windows Server 2008 Enterprise", 1, 10
        )
        config["DC"]["osProbability"]["Windows Server 2008 R2 Standard"] = trial.suggest_int(
            "DC_Windows Server 2008 R2 Standard", 1, 20
        )
        config["DC"]["osProbability"]["Windows Server 2008 R2 Datacenter"] = trial.suggest_int(
            "DC_Windows Server 2008 R2 Datacenter", 1, 20
        )
        config["DC"]["osProbability"]["Windows Server 2008 R2 Enterprise"] = trial.suggest_int(
            "DC_Windows Server 2008 R2 Enterprise", 1, 20
        )
        config["DC"]["osProbability"]["Windows Server 2012 Standard"] = trial.suggest_int(
            "DC_Windows Server 2012 Standard", 1, 30
        )
        config["DC"]["osProbability"]["Windows Server 2012 Datacenter"] = trial.suggest_int(
            "DC_Windows Server 2012 Datacenter", 1, 30
        )
        config["DC"]["osProbability"]["Windows Server 2012 R2 Standard"] = trial.suggest_int(
            "DC_Windows Server 2012 R2 Standard", 1, 40
        )
        config["DC"]["osProbability"]["Windows Server 2012 R2 Datacenter"] = trial.suggest_int(
            "DC_Windows Server 2012 R2 Datacenter", 1, 40
        )
        config["DC"]["osProbability"]["Windows Server 2016 Standard"] = trial.suggest_int(
            "DC_Windows Server 2016 Standard", 1, 50
        )
        config["DC"]["osProbability"]["Windows Server 2016 Datacenter"] = trial.suggest_int(
            "DC_Windows Server 2016 Datacenter", 1, 50
        )

        config["User"]["nUsers"] = trial.suggest_int("User_nUsers", 50, 1000)
        config["User"]["enabled"] = trial.suggest_int("User_enabled", 1, 100)
        config["User"]["dontreqpreauth"] = trial.suggest_int("User_dontreqpreauth", 1, 50)
        config["User"]["hasspn"] = trial.suggest_int("User_hasspn", 1, 50)
        config["User"]["passwordnotreqd"] = trial.suggest_int("User_passwordnotreqd", 1, 50)
        config["User"]["pwdneverexpires"] = trial.suggest_int("User_pwdneverexpires", 1, 100)
        config["User"]["sidhistory"] = trial.suggest_int("User_sidhistory", 1, 50)
        config["User"]["unconstraineddelegation"] = trial.suggest_int(
            "User_unconstraineddelegation", 1, 50
        )
        config["User"]["savedcredentials"] = trial.suggest_int("User_savedcredentials", 1, 50)

        config["OU"]["nOUs"] = trial.suggest_int("OU_nOUs", 10, 200)

        config["GPO"]["nGPOs"] = trial.suggest_int("GPO_nGPOs", 10, 100)
        config["GPO"]["exploitable"] = trial.suggest_int("GPO_exploitable", 1, 50)

        config["Group"]["nGroups"] = trial.suggest_int("Group_nGroups", 50, 200)
        config["Group"]["nestingGroupProbability"] = trial.suggest_int(
            "Group_nestingGroupProbability", 1, 50
        )
        config["Group"]["departmentProbability"]["IT"] = trial.suggest_int("Group_IT", 1, 50)
        config["Group"]["departmentProbability"]["HR"] = trial.suggest_int("Group_HR", 1, 50)
        config["Group"]["departmentProbability"]["MARKETING"] = trial.suggest_int(
            "Group_MARKETING", 1, 50
        )
        config["Group"]["departmentProbability"]["OPERATIONS"] = trial.suggest_int(
            "Group_OPERATIONS", 1, 50
        )
        config["Group"]["departmentProbability"]["BIDNESS"] = trial.suggest_int(
            "Group_BIDNESS", 1, 50
        )

        config["ACLs"]["ACLPrincipalsPercentage"] = trial.suggest_int(
            "ACLs_ACLPrincipalsPercentage", 1, 100
        )
        config["ACLs"]["ACLsProbability"]["GenericAll"] = trial.suggest_int(
            "ACLs_GenericAll", 1, 50
        )
        config["ACLs"]["ACLsProbability"]["GenericWrite"] = trial.suggest_int(
            "ACLs_GenericWrite", 1, 50
        )
        config["ACLs"]["ACLsProbability"]["WriteOwner"] = trial.suggest_int(
            "ACLs_WriteOwner", 1, 50
        )
        config["ACLs"]["ACLsProbability"]["WriteDacl"] = trial.suggest_int("ACLs_WriteDacl", 1, 50)
        config["ACLs"]["ACLsProbability"]["AddMember"] = trial.suggest_int("ACLs_AddMember", 1, 50)
        config["ACLs"]["ACLsProbability"]["ForceChangePassword"] = trial.suggest_int(
            "ACLs_ForceChangePassword", 1, 50
        )
        config["ACLs"]["ACLsProbability"]["ReadLAPSPassword"] = trial.suggest_int(
            "ACLs_ReadLAPSPassword", 1, 50
        )

        try:
            env.parameters = config
            env.generate()
            return True
        except Exception as error:
            print(f"[!] Failed to generate! {error}")
            return False

    return objective


def task_generate_environment(url, username, password, domain):
    print("Generating environment")
    print(f"URL: {url}")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"Domain: {domain}")

    env = MainMenu(url=url, username=username, password=password, domain=domain)

    try:
        env.connect()
        print("[+] Successfully connected!")
    except Exception as error:
        print(f"[!] Failed to connect! {error}")

    try:
        env.generate()
        print("[+] Done!")
    except Exception as error:
        print(f"[!] Failed to generate! {error}")


if __name__ == "__main__":
    fire.Fire(task_generate_environment)
